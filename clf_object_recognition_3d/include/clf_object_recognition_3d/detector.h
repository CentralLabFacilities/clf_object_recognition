#pragma once

#include "ros/ros.h"
#include <ros/node_handle.h>     // for NodeHandle
#include <ros/service_server.h>  // for ServiceServer
#include <ros/service_client.h>

#include <dynamic_reconfigure/server.h>
#include <clf_object_recognition_cfg/Detect3dConfig.h>

#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>

// message types in
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <vision_msgs/BoundingBox2D.h>
#include <vision_msgs/BoundingBox3D.h>
#include <vision_msgs/Detection3DArray.h>

// message types out
#include <sensor_msgs/PointCloud2.h>
#include <clf_object_recognition_msgs/Detect3D.h>

// point cloud types
#include <pcl/common/io.h>

#include <mutex>

typedef pcl::PointXYZ point_type;
typedef pcl::PointCloud<point_type> pointcloud_type;

class Detector {
public:
    Detector(ros::NodeHandle nh);
private:
    void ReconfigureCallback(const clf_object_recognition_cfg::Detect3dConfig& config, uint32_t level);
    void Callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth_image, const sensor_msgs::CameraInfoConstPtr& camera_info);
    bool ServiceDetect3D(clf_object_recognition_msgs::Detect3D::Request& req,  clf_object_recognition_msgs::Detect3D::Response& res);
    pointcloud_type* createPointCloudFromDepthImage(const sensor_msgs::Image& depth_msg, const vision_msgs::BoundingBox2D& bbox, const sensor_msgs::CameraInfoConstPtr& cam_info);
    pointcloud_type* createPointCloudFromMesh(const std::string& mesh_name);

    ros::NodeHandle nh_;
    clf_object_recognition_cfg::Detect3dConfig config;
    dynamic_reconfigure::Server<clf_object_recognition_cfg::Detect3dConfig> reconfigure_server;
    ros::ServiceServer srv_detect_3d;
    ros::ServiceClient srv_detect_2d;

    // incoming messages
    sensor_msgs::Image::ConstPtr image_;
    sensor_msgs::Image::ConstPtr depth_image_;
    sensor_msgs::CameraInfo::ConstPtr camera_info_;

    // subscribers
    message_filters::Subscriber<sensor_msgs::Image> image_sub_;
    message_filters::Subscriber<sensor_msgs::Image> depth_image_sub_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_sub_;

    // publisher
    ros::Publisher pub_detections_3d;
    ros::Publisher pub_raw_pcl;

    // sync with exact policy
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo> sync_;

    std::mutex mutex_;
};