#pragma once

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

#include "clf_object_recognition_msgs/Img2RawPointCloudMsg.h"

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
        clf_object_recognition_msgs::Img2RawPointCloudMsg::Request& req,
        clf_object_recognition_msgs::Img2RawPointCloudMsg::Response& res);

int main(int argc, char** argv);


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