#include "clf_object_recognition_3d/detector.h"
#include "clf_object_recognition_msgs/Detect2DImage.h"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"
//#include "pcl_ros/io/pcl_conversions.h"
#include <pcl_conversions/pcl_conversions.h>

#include <ros/console.h>

Detector::Detector(ros::NodeHandle nh)
    : sync_(image_sub_, depth_image_sub_, camera_info_sub_, 10) 
 {
    srv_detect_2d = nh.serviceClient<clf_object_recognition_msgs::Detect2DImage>("/yolox/recognize_from_image");
    srv_detect_3d = nh.advertiseService("simple_detections", &Detector::ServiceDetect3D, this);

    // del me later
    pub_detections_3d = nh.advertise<vision_msgs::Detection3DArray>("last_detection", 1);
    pub_raw_pcl = nh.advertise<sensor_msgs::PointCloud2>("raw_pcl", 1);

    //vision_msgs::Detection3DArray

    // subscribe to camera topics
    image_sub_.subscribe(nh, "/xtion/rgb/image_raw", 1);
    depth_image_sub_.subscribe(nh, "/xtion/depth_registered/image_raw", 1);
    camera_info_sub_.subscribe(nh, "/xtion/depth_registered/camera_info", 1);

    // sync incoming camera messages
    sync_.registerCallback(boost::bind(&Detector::Callback, this, _1, _2, _3));

    auto f = [this](auto&& PH1, auto&& PH2) { ReconfigureCallback(PH1, PH2); };
    reconfigure_server.setCallback(f);
    ros::spinOnce();
}

void Detector::Callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth_image, const sensor_msgs::CameraInfoConstPtr& camera_info) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    image_ = image;
    depth_image_ = depth_image;
    camera_info_ = camera_info;
}

void Detector::ReconfigureCallback(const clf_object_recognition_cfg::Detect3dConfig& input, uint32_t /*level*/)
{
    ROS_INFO_NAMED("detector", "Reconfigure");
    config = input;
}
    
bool Detector::ServiceDetect3D(clf_object_recognition_msgs::Detect3D::Request& req, clf_object_recognition_msgs::Detect3D::Response& res) {
    sensor_msgs::Image img;
    sensor_msgs::Image depth;
    sensor_msgs::CameraInfo info;
    
    {
        std::lock_guard<std::mutex> lock(mutex_);
        img = sensor_msgs::Image(*image_.get());
        depth = sensor_msgs::Image(*depth_image_.get());
    }
  

    ROS_INFO_STREAM_NAMED("detector", "ServiceDetect3D() called " << req);

    clf_object_recognition_msgs::Detect2DImage param;
    param.request.image = img;

    auto ok = srv_detect_2d.call(param);
    if(!ok) {
        ROS_ERROR_STREAM_NAMED("detector", "cant call detections ");
        return false;
    }

    
    ROS_INFO_STREAM_NAMED("detector", "got detections " << param.response);
    for(auto& detection : param.response.detections) {
        vision_msgs::Detection3D d3d;
        // generate point cloud from incoming depth image for detection bounding box
        pointcloud_type* cloud_from_depth_image = createPointCloudFromDepthImage(depth, detection.bbox, camera_info_);
        // pointcloud_type* cloud_from_mesh = createPointCloudFromMesh(mesh_name);
        
        Eigen::Vector4d cloud_from_depth_image_centroid = Eigen::Vector4d::Random();
        auto centroid_size = pcl::compute3DCentroid(*cloud_from_depth_image, cloud_from_depth_image_centroid);

        sensor_msgs::PointCloud2 pcl_msg;
        pcl::toROSMsg(*cloud_from_depth_image, pcl_msg);
        pub_raw_pcl.publish(pcl_msg);

        d3d.header = detection.header;
        d3d.bbox.center.orientation.w = 1;
        d3d.bbox.center.position.x = cloud_from_depth_image_centroid.x();
        d3d.bbox.center.position.y = cloud_from_depth_image_centroid.y();
        d3d.bbox.center.position.z = cloud_from_depth_image_centroid.z();
        d3d.bbox.size.x = 0.1;
        d3d.bbox.size.y = 0.1;
        d3d.bbox.size.z = 0.1;

        for(auto hypo : detection.results) {
           vision_msgs::ObjectHypothesisWithPose hyp;
           hyp = hypo;
           hyp.pose.pose.orientation.w = 1;

           d3d.results.push_back(hyp); 
        }

        res.detections.push_back(d3d);
    }

    
    if(config.publish_detections) {
        ROS_INFO_STREAM_NAMED("detector", "publishing detections ");
        vision_msgs::Detection3DArray a;
        for(auto b : res.detections)  {
            a.header = b.header;
            a.detections.push_back(b);
        }
        pub_detections_3d.publish(a);
    }

    

    return true;

}

pointcloud_type* Detector::createPointCloudFromMesh(const std::string& mesh_name) {
    pointcloud_type* cloud (new pointcloud_type());
    // generate point cloud
    return cloud;
}

pointcloud_type* Detector::createPointCloudFromDepthImage(const sensor_msgs::Image& depth_msg, const vision_msgs::BoundingBox2D& bbox, const sensor_msgs::CameraInfoConstPtr& cam_info) 
{
    pointcloud_type* cloud (new pointcloud_type());

    cloud->header.stamp     = ros::Time(depth_msg.header.stamp).toSec();
    cloud->header.frame_id  = depth_msg.header.frame_id;
    //single point of view, 2d rasterized
    cloud->is_dense         = true; 
    
    //principal point and focal lengths
    float cx, cy, fx, fy;
    
    cloud->height = bbox.size_y;
    cloud->width = bbox.size_x;
    cx = cam_info->K[2]; //(cloud->width >> 1) - 0.5f;
    cy = cam_info->K[5]; //(cloud->height >> 1) - 0.5f;
    fx = 1.0f / cam_info->K[0]; 
    fy = 1.0f / cam_info->K[4]; 
    
    cloud->points.resize (bbox.size_y * bbox.size_x);
    
    const float* depth_buffer = reinterpret_cast<const float*>(&depth_msg.data[0]);

    int depth_idx = 0;
    
    pointcloud_type::iterator pt_iter = cloud->begin();
    for (int v = bbox.center.y - bbox.size_y/2; v < bbox.size_y; ++v)
    {
        for (int u = bbox.center.x - bbox.size_x / 2; u < bbox.size_x; ++u)
        {
            point_type& pt = *pt_iter++;
            float Z = depth_buffer[depth_idx++];
        
            // Check for invalid measurements
            if (std::isnan(Z))
            {
                pt.x = pt.y = pt.z = Z;
            }
            else // Fill in XYZ
            {
                pt.x = (u - cx) * Z * fx;
                pt.y = (v - cy) * Z * fy;
                pt.z = Z;
            }

            ROS_INFO_STREAM_NAMED("Detector ", "Points: " << pt.x << ',' << pt.y << ',' << pt.z);
        }
    }
    
    return cloud;
}
