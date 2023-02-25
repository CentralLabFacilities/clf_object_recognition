#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <your_package_name/RegisterPointClouds.h>

bool registerPointClouds(your_package_name::RegisterPointClouds::Request& req,
                         your_package_name::RegisterPointClouds::Response& res)
{
  // Load the raw point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(req.raw_cloud, *raw_cloud);

  // Load the reference point cloud
  pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::fromROSMsg(req.ref_cloud, *ref_cloud);

  // Apply voxel grid filter to the reference point cloud
  pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
  voxel_grid.setLeafSize(0.01, 0.01, 0.01);
  pcl::PointCloud<pcl::PointXYZ>::Ptr ref_cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
  voxel_grid.setInputCloud(ref_cloud);
  voxel_grid.filter(*ref_cloud_filtered);

  // Register the reference point cloud to the raw point cloud
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setInputSource(ref_cloud_filtered);
  icp.setInputTarget(raw_cloud);
  pcl::PointCloud<pcl::PointXYZ> registered_cloud;
  icp.align(registered_cloud);

  // Convert the registered point cloud to a ROS message and publish it
  sensor_msgs::PointCloud2 registered_cloud_msg;
  pcl::toROSMsg(registered_cloud, registered_cloud_msg);
  registered_cloud_msg.header = req.raw_cloud.header;
  res.registered_cloud = registered_cloud_msg;

  return true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "point_cloud_registration_service");
  ros::NodeHandle nh;

  ros::ServiceServer service = nh.advertiseService("register_point_clouds", registerPointClouds);
  ROS_INFO("Ready to register point clouds.");
  ros::spin();

  return 0;
}

/*
# Service request message
sensor_msgs/PointCloud2 raw_cloud     # Raw point cloud to register (ROS PointCloud2 message)
sensor_msgs/PointCloud2 ref_cloud     # Reference point cloud for registration (ROS PointCloud2 message)

---
# Service response message
sensor_msgs/PointCloud2 registered_cloud  # Clean, registered point cloud (ROS PointCloud2 message)

*/