#include "clf_object_recognition_3d/cloud_from_image.h"
#include "sensor_msgs/image_encodings.h"
#include <cv_bridge/cv_bridge.h>

#include <ros/console.h>

namespace cloud
{
pcl::PointCloud<pcl::PointXYZ>::Ptr fromDepthImage(const sensor_msgs::Image& depth, sensor_msgs::CameraInfo info,
                                                   double depth_scaling)
{
  image_geometry::PinholeCameraModel camera;
  camera.fromCameraInfo(info);

  // todo encodings from image
  cv_bridge::CvImagePtr cv_ptr;
  if (depth.encoding == sensor_msgs::image_encodings::MONO8)
  {
    cv_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_8UC1);
  }
  else
  {
    cv_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_16UC1);
  }

  float constant_x = depth_scaling / camera.fx();
  float constant_y = depth_scaling / camera.fy();

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

  cloud->header.stamp = ros::Time(depth.header.stamp).toSec();
  cloud->header.frame_id = depth.header.frame_id;
  cloud->is_dense = true;

  int w = cv_ptr->image.cols;
  int h = cv_ptr->image.rows;

  cloud->points.resize(w * h);

  for (int u = 0; u < w; u++)
  {
    for (int v = 0; v < h; v++)
    {
      float depth = cv_ptr->image.at<uint16_t>(v, u);
      auto& pt = cloud->points[v * w + u];
      if (depth != 0)
      {
        pt.x = (u - camera.cx()) * depth * constant_x;
        pt.y = (v - camera.cy()) * depth * constant_y;
        pt.z = depth * depth_scaling;
      }
      else
      {
        cloud->is_dense = false;
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }

  cloud->width = w;
  cloud->height = h;

  return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr fromDepthArea(const vision_msgs::BoundingBox2D& bbox,
                                                  const sensor_msgs::Image& depth, sensor_msgs::CameraInfo info,
                                                  double depth_scaling)
{
  ROS_DEBUG_STREAM_NAMED("cloud", "fromDepthArea " << );
  image_geometry::PinholeCameraModel camera;
  camera.fromCameraInfo(info);

  // todo encodings from image
  cv_bridge::CvImagePtr cv_ptr;
  if (depth.encoding == sensor_msgs::image_encodings::MONO8)
  {
    cv_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_8UC1);
  }
  else
  {
    cv_ptr = cv_bridge::toCvCopy(depth, sensor_msgs::image_encodings::TYPE_16UC1);
  }
  ROS_DEBUG_STREAM_NAMED("cloud", "copied image");

  float constant_x = depth_scaling / camera.fx();
  float constant_y = depth_scaling / camera.fy();

  int w = bbox.size_x;
  int h = bbox.size_y;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

  cloud->header.stamp = ros::Time(depth.header.stamp).toSec();
  cloud->header.frame_id = depth.header.frame_id;
  cloud->is_dense = true;

  cloud->points.resize(w * h);

  ROS_DEBUG_STREAM_NAMED("cloud", "image size: " << cv_ptr->image.cols << "x" << cv_ptr->image.rows;);
  ROS_DEBUG_STREAM_NAMED("cloud", "box size: " << w << "x" << h);
  ROS_DEBUG_STREAM_NAMED("cloud", "box center: " << bbox.center.x << "," << bbox.center.y);

  ROS_DEBUG_STREAM_NAMED("cloud", "loop");
  for (int u = (int)(bbox.center.x - bbox.size_x / 2); u < (int)(bbox.center.x + bbox.size_x / 2); u++)
  {
    for (int v = (int)(bbox.center.y - bbox.size_y / 2); v < (int)(bbox.center.y + bbox.size_y / 2); v++)
    {
      float depth = cv_ptr->image.at<uint16_t>(v, u);
      auto& pt = cloud->points[v * w + u];
      if (depth != 0)
      {
        pt.x = (u - camera.cx()) * depth * constant_x;
        pt.y = (v - camera.cy()) * depth * constant_y;
        pt.z = depth * depth_scaling;
      }
      else
      {
        cloud->is_dense = false;
        pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
      }
    }
  }

  cloud->width = w;
  cloud->height = h;

  return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr oldFromDepth(const sensor_msgs::Image& depth_msg, const vision_msgs::BoundingBox2D& bbox,
                                  const sensor_msgs::CameraInfoConstPtr& cam_info)
{
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

  cloud->header.stamp = ros::Time(depth_msg.header.stamp).toSec();
  cloud->header.frame_id = depth_msg.header.frame_id;
  ROS_INFO_STREAM_NAMED("Detector ", "frame_id: " << depth_msg.header.frame_id);

  cloud->is_dense = true;

  image_geometry::PinholeCameraModel camera;
  camera.fromCameraInfo(cam_info);

  // principal point and focal lengths
  float cx, cy, fx, fy;

  // cloud->height = depth_msg.height;
  // cloud->width = depth_msg.width;
  cx = cam_info->K[2];  //(cloud->width >> 1) - 0.5f;
  cy = cam_info->K[5];  //(cloud->height >> 1) - 0. f;
  fx = 1.0f / cam_info->K[0];
  fy = 1.0f / cam_info->K[4];

  // cloud->points.resize (cloud->height * cloud->width);
  cloud->points.resize(bbox.size_x * bbox.size_y);

  const float* depth_buffer = reinterpret_cast<const float*>(&depth_msg.data[0]);

  int depth_idx = 0;

  pcl::PointCloud<pcl::PointXYZ>::iterator pt_iter = cloud->begin();
  for (int v = (int)(bbox.center.y - bbox.size_y / 2); v < (int)(bbox.center.y - bbox.size_y / 2 + bbox.size_y); ++v)
  {
    for (int u = (int)(bbox.center.x - bbox.size_x / 2); u < (int)(bbox.center.x - bbox.size_x / 2 + bbox.size_x);
         ++u, ++pt_iter)
    {
      auto& pt = *pt_iter;
      depth_idx = depth_msg.width * v + u;
      float Z = depth_buffer[depth_idx];

      // Check for invalid measurements
      if (std::isnan(Z))
      {
        pt.x = pt.y = pt.z = Z;
      }
      else  // Fill in XYZ
      {
        pt.x = (u - cx) * Z * fx;
        pt.y = (v - cy) * Z * fy;
        pt.z = Z;
      }
    }
  }

  return cloud;
}

}  // namespace cloud