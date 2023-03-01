#pragma once

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

#include "clf_object_recognition_msgs/PclToDetectionProvider.h"

vision_msgs::Detection3D pcl_to_detection(const sensor_msgs::PointCloud2& pcl_msg, 
                                          const std::string& class_name, const float& score);


bool detection_service(vision_msgs::DetectObject::Request& req,
                       vision_msgs::DetectObject::Response& res);

int main(int argc, char** argv);


/*
sensor_msgs/PointCloud2 pcl
string class_name
*/