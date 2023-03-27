#pragma once

#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>
// #include <your_package/LoadPointCloud.h>
#include "clf_object_recognition_msgs/LoadPointCloudMsg.h"

bool loadPointCloud(clf_object_recognition_msgs::LoadPointCloudMsg::Request& req,
                    clf_object_recognition_msgs::LoadPointCloudMsg::Response& res);

int main(int argc, char** argv);