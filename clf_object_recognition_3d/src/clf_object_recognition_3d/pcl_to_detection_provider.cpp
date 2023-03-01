#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>

//#include <object_msgs/Detection3D.h>
//#include <object_msgs/ObjectHypothesis.h>
#include <vision_msgs/Detection3D.h>
#include <vision_msgs/ObjectHypothesis.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

vision_msgs::Detection3D pcl_to_detection(const sensor_msgs::PointCloud2& pcl_msg, const std::string& class_name, const float& score)
{
    vision_msgs::Detection3D detection;
    // convert pcl_msg to a pcl::PointCloud<pcl::PointXYZ> object
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::fromROSMsg(pcl_msg, cloud);

    // set the results field based on the object classification
    vision_msgs::ObjectHypothesis hypothesis;
    hypothesis.class_id = class_name;
    hypothesis.score = score;
    detection.results.push_back(hypothesis);

    // set the bounding box field based on the point cloud
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(cloud, min_pt, max_pt);
    geometry_msgs::Vector3 size;
    size.x = max_pt[0] - min_pt[0];  // size of the bounding box in x direction
    size.y = max_pt[1] - min_pt[1];  // size of the bounding box in y direction
    size.z = max_pt[2] - min_pt[2];  // size of the bounding box in z direction
    detection.bbox.size = size;

    geometry_msgs::Pose pose;
    pose.position.x = (min_pt[0] + max_pt[0]) / 2.0;  // x position of the center of the bounding box
    pose.position.y = (min_pt[1] + max_pt[1]) / 2.0;  // y position of the center of the bounding box
    pose.position.z = (min_pt[2] + max_pt[2]) / 2.0;  // z position of the center of the bounding box
    pose.orientation.w = 1.0;  // quaternion representing the orientation of the object
    detection.bbox.center = pose;

    // set the pose field based on the point cloud
    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(cloud, centroid);
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.frame_id = pcl_msg.header.frame_id;
    pose_stamped.pose = pose;
    detection.pose = pose_stamped;

    return detection;
}

// bool detection_service(vision_msgs::DetectObject::Request& req,
//                       vision_msgs::DetectObject::Response& res)

bool detection_service(vision_msgs::DetectObject::Request& req,
                       vision_msgs::DetectObject::Response& res)
{
    vision_msgs::Detection3D detection = pcl_to_detection(req.pcl, req.class_name, req.score);
    res.detection = detection;
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "detection_service");
    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("detect_object", detection_service);
    ROS_INFO("Ready to detect objects.");

    ros::spin();

    return 0;
}


/*
sensor_msgs/PointCloud2 pcl
string class_name
*/