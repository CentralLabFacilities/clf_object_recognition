#pragma once

#include "ros/ros.h"
#include <ros/node_handle.h>     // for NodeHandle
#include <ros/service_server.h>  // for ServiceServer
#include <ros/service_client.h>

#include <tf2_ros/transform_listener.h>

// message types in
#include <sensor_msgs/PointCloud2.h>
#include <clf_object_recognition_msgs/GetFloat.h>
#include <clf_object_recognition_msgs/Door.h>

// pcl types
#include <pcl/PolygonMesh.h>
#include <pcl/TextureMesh.h>
#include <pcl/common/io.h>
#include <pcl/filters/crop_box.h>

#include <memory>

typedef pcl::PointXYZ point_t;

class Detector
{
public:
  Detector(ros::NodeHandle nh);

private:
  bool ServiceDetectTable(clf_object_recognition_msgs::GetFloat::Request& req,
                        clf_object_recognition_msgs::GetFloat::Response& res);
  bool ServiceIsDoorOpen(clf_object_recognition_msgs::Door::Request& req,
                        clf_object_recognition_msgs::Door::Response& res);

  ros::NodeHandle nh_;

  ros::ServiceServer srv_table_;

  // publisher
  ros::Publisher pub_cloud_table_;

  tf2_ros::Buffer tf_buffer;
  tf2_ros::TransformListener tf_listener;
};