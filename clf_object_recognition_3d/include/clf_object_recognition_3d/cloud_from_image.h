#pragma once

#include <pcl/common/io.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <vision_msgs/BoundingBox2D.h>

#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>

namespace cloud
{
pcl::PointCloud<pcl::PointXYZ>::Ptr fromDepthImage(const sensor_msgs::Image& depth, sensor_msgs::CameraInfo info,
                                                   double depth_scaling = 0.001F);
pcl::PointCloud<pcl::PointXYZ>::Ptr fromDepthArea(const vision_msgs::BoundingBox2D& bbox,
                                                  const sensor_msgs::Image& depth, sensor_msgs::CameraInfo info,
                                                  double depth_scaling = 0.001F);
pcl::PointCloud<pcl::PointXYZ>::Ptr oldFromDepth(const sensor_msgs::Image& depth_msg,
                                                 const vision_msgs::BoundingBox2D& bbox,
                                                 const sensor_msgs::CameraInfoConstPtr& cam_info);

}  // namespace cloud