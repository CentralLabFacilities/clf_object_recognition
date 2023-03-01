#pragma once

#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
// #include <your_package_name/RegisterPointClouds.h>
//#include "clf_object_recognition_msgs/RegistratedPclProvider.h"

bool registerPointClouds(your_package_name::RegisterPointClouds::Request& req,
                         your_package_name::RegisterPointClouds::Response& res);

int main(int argc, char** argv);

/*
# Service request message
sensor_msgs/PointCloud2 raw_cloud     # Raw point cloud to register (ROS PointCloud2 message)
sensor_msgs/PointCloud2 ref_cloud     # Reference point cloud for registration (ROS PointCloud2 message)

---
# Service response message
sensor_msgs/PointCloud2 registered_cloud  # Clean, registered point cloud (ROS PointCloud2 message)

*/