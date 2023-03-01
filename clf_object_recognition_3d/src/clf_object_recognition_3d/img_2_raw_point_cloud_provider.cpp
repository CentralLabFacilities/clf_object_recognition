#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <cv_bridge/cv_bridge.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

// #include <clf_object_recognition_msgs/Img2RawPointCloudProvider.h>

/*
Service Name:
    pointcloud_from_depth_image

Service Type:
    object_detection/GetPointCloud

Description:
    Generates a point cloud from an RGB image, depth image, and camera intrinsic parameters, cropped to a specified 2D bounding box.
    The output point cloud is filtered to remove statistical outliers and downsampled using a voxel grid filter.

Inputs:
    string camera_link: Name of camera's frame ID
    string fixed_frame: Name of fixed frame ID
    sensor_msgs/Image image: RGB image
    sensor_msgs/Image depth_image: Depth image
    sensor_msgs/CameraInfo camera_info: Camera intrinsic parameters
    int32 xmin: Minimum x-coordinate of bounding box
    int32 ymin: Minimum y-coordinate of bounding box
    int32 xmax: Maximum x-coordinate of bounding box
    int32 ymax: Maximum y-coordinate of bounding box
    string class_name: Object class name
    float32 certainty: Object detection certainty score

Outputs:
    bool success: True if processing was successful
    string class_name: Object class name
    float32 certainty: Object detection certainty score
    object_detection/BoundingBox2D bbox: 2D bounding box coordinates
    sensor_msgs/PointCloud2

*/


bool pointcloud_from_depth_image_service_callback(
        sensor_msgs::PointCloud2::Request& req,
        sensor_msgs::PointCloud2::Response& res) {

    // Extract inputs from request
    cv_bridge::CvImagePtr img = cv_bridge::toCvCopy(req.image, sensor_msgs::image_encodings::BGR8);
    cv_bridge::CvImagePtr depth_img = cv_bridge::toCvCopy(req.depth_image, sensor_msgs::image_encodings::TYPE_16UC1);
    sensor_msgs::CameraInfo info_msg = req.camera_info;
    std::string class_name = req.class_name;
    float certainty = req.certainty;

    // Check if bounding box is specified
    bool is_bbox_specified = (req.bounding_box.xmin != 0 || req.bounding_box.ymin != 0 ||
                              req.bounding_box.xmax != 0 || req.bounding_box.ymax != 0);

    // Extract bounding box coordinates if specified
    int xmin = 0, ymin = 0, xmax = img->image.cols, ymax = img->image.rows;
    if (is_bbox_specified) {
        int xmin = req.bounding_box.xmin;
        int ymin = req.bounding_box.ymin;
        int xmax = req.bounding_box.xmax;
        int ymax = req.bounding_box.ymax;
    }

    // Extract bounding box coordinates
    float xc = (xmin + xmax) / 2.0;
    float yc = (ymin + ymax) / 2.0;
    float width = xmax - xmin;
    float height = ymax - ymin;

    // Extract camera transformation from the transform tree
    tf2_ros::Buffer tf_buffer;
    tf2_ros::TransformListener tf_listener(tf_buffer);
    geometry_msgs::TransformStamped transformStamped;
    try{
        transformStamped = tf_buffer.lookupTransform(req.camera_link, req.fixed_frame, ros::Time(0));
    }
    catch (tf2::TransformException &ex) {
        ROS_WARN("%s",ex.what());
        return false;
    }

    // Crop RGB and depth images
    cv::Mat img_mat = img->image;
    cv::Mat depth_mat = depth_img->image;
    if (is_bbox_specified) {
        cv::Mat rgb_crop = img_mat(cv::Range(ymin, ymax), cv::Range(xmin, xmax));
        cv::Mat depth_crop = depth_mat(cv::Range(ymin, ymax), cv::Range(xmin, xmax));
        img_mat = rgb_crop;
        depth_mat = depth_crop;
    }

    // cv::Mat rgb_crop = img_mat(cv::Range(ymin, ymax), cv::Range(xmin, xmax)); 
    // cv::Mat depth_crop = depth_mat(cv::Range(ymin, ymax), cv::Range(xmin, xmax));

    // Convert depth image to point cloud
    // The crop region is defined in the camera frame, but the resulting point cloud needs to be transformed to the fixed frame
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointXYZ point;
    point.z = 0;
    for (int r = 0; r < depth_crop.rows; r++) {
        for (int c = 0; c < depth_crop.cols; c++) {
            ushort depth_val = depth_crop.at<ushort>(r, c);
            if (depth_val != 0) {
                cv::Point3f pt = depthTo3D(cv::Point2i(c + xmin, r + ymin), depth_val, info_msg);
                point.x = pt.x;
                point.y = pt.y;
                pcl_cloud->points.push_back(point);
            }
        }
    }
    pcl_cloud->width = pcl_cloud->points.size();
    pcl_cloud->height = 1;

    // Transform point cloud to fixed frame
    // pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    // Eigen::Affine3d transform_eigen;
    // tf::transformMsgToEigen(transformStamped.transform, transform_eigen);
    //pcl::transformPointCloud(*pcl_cloud, *pcl_transformed_cloud, transform_eigen);

    // Apply voxel grid filter
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
    voxel_grid.setInputCloud(pcl_cloud);
    //voxel_grid.setInputCloud(pcl_transformed_cloud);
    voxel_grid.setLeafSize(0.01f, 0.01f, 0.01f);
    voxel_grid.filter(*pcl_filtered);

    // Convert filtered point cloud to PCL XYZRGB format
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*pcl_filtered, *pcl_colored_cloud);

    // Apply statistical outlier removal filter
    pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
    sor.setInputCloud(pcl_colored_cloud);
    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);
    sor.filter(*pcl_colored_cloud);

    // Convert colored point cloud back to XYZ format
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_outliers_removed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*pcl_colored_cloud, *pcl_outliers_removed_cloud);

    // Create the response message
    object_detection::GetPointCloudResponse response;
    response.success = true;
    response.class_name = class_name;
    response.certainty = certainty;
    response.bbox.xmin = xmin;
    response.bbox.ymin = ymin;
    response.bbox.xmax = xmax;
    response.bbox.ymax = ymax;

    // Convert PCL point cloud to ROS point cloud message
    sensor_msgs::PointCloud2 pcl_output;
    pcl::toROSMsg(*pcl_outliers_removed_cloud, pcl_output);
    // pcl_output.header.frame_id = "camera_link";  // Replace "camera_link" with your camera's frame ID
    pcl_output.header.frame_id = req.camera_link;
    response.pointcloud = pcl_output;

    // Compute the transform from the camera to the cropped region
    tf::StampedTransform transform;
    try
        {
            //tf_listener.lookupTransform("camera_link", img->header.frame_id, ros::Time(0), transform);
            tf_listener.lookupTransform(req.camera_link, req.fixed_frame, ros::Time(0), transform);
        }
    catch (tf::TransformException& ex)
        {
            ROS_ERROR_STREAM("Error getting transform: " << ex.what());
            response.success = false;
            return false;
        }
    tf::Transform crop_transform;
    crop_transform.setOrigin(
        tf::Vector3(xmin * info_msg.K[0] + info_msg.K[2], ymin * info_msg.K[4] + info_msg.K[5], 0));
    crop_transform.setRotation(tf::Quaternion::getIdentity());
    transform = transform * crop_transform;
    tf::transformStampedTFToMsg(transform, response.transform);

    // Send the response message
    res = response;
    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "img_2_raw_point_cloud_service");
    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("img_2_raw_point_cloud", pointcloud_from_depth_image_service_callback);

    ROS_INFO("Point Cloud service ready.");
    ros::spin();

    return 0;
}


/*
Example usage:

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from object_detection.srv import GetPointCloud

rospy.wait_for_service('pointcloud_from_depth_image')

try:
    get_pointcloud = rospy.ServiceProxy('pointcloud_from_depth_image', GetPointCloud)
    resp = get_pointcloud(camera_link='camera_link', fixed_frame='fixed_frame', image=image_msg,
                           depth_image=depth_msg, camera_info=camera_info_msg, xmin=10, ymin=20,
                           xmax=100, ymax=200, class_name='person', certainty=0.9)
    if resp.success:
        print(f"Point cloud for {resp.class_name} detected with certainty {resp.certainty}:\n{resp.pointcloud}")
    else:
        print("Point cloud generation failed.")
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)

*/


/*
// Service message

string fixed_frame  # Name of fixed frame ID
string camera_link  # Name of camera's frame ID
sensor_msgs/Image image  # RGB image
sensor_msgs/Image depth_image  # Depth image
sensor_msgs/CameraInfo camera_info  # Camera intrinsic parameters
int32 xmin  # Minimum x-coordinate of bounding box
int32 ymin  # Minimum y-coordinate of bounding box
int32 xmax  # Maximum x-coordinate of bounding box
int32 ymax  # Maximum y-coordinate of bounding box
string class_name  # Object class name
float32 certainty  # Object detection score
---
bool success  # True if processing was successful
string class_name  # Object class name
float32 certainty  # Object detection certainty score
object_detection/BoundingBox2D bbox  # 2D bounding box coordinates
sensor_msgs/PointCloud2 pointcloud  # Output point cloud in ROS message format

*/