#pragma once

#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
// #include <your_package/LoadPointCloud.h>
#include "clf_object_recognition_msgs/LoadPointCloudProvider.h"


bool loadPointCloud(clf_object_recognition_msgs::LoadPointCloudProvider::Request& req,
                    clf_object_recognition_msgs::LoadPointCloudProvider::Response& res);

int main(int argc, char** argv);