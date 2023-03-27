#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
// #include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/point_types.h>
#include <sensor_msgs/PointCloud2.h>

/**

@brief Converts a mesh file to a point cloud, samples it, and saves the point cloud to a PCD file

@param req Request object containing source file path, target file path, and number of points to sample

@param res Response object containing the converted point cloud and number of points sampled

@return true if the conversion is successful, false otherwise

@note The source file should be in a format supported by PCL (e.g. .ply, .obj, .stl).

@note The target file should have the .pcd file extension.
*/
bool convertMeshToPclMsg(sensor_msgs::PointCloud2::Request& req, sensor_msgs::PointCloud2::Response& res)
{
  // Get the parameters from the request
  std::string source_file = req.source_file;
  std::string target_file = req.target_file;
  int num_points = req.num_points;

  // Load the mesh file
  pcl::PolygonMesh mesh;
  pcl::io::loadPolygonFile(source_file, mesh);

  // Convert the mesh to a point cloud
  pcl::PointCloudpcl::PointXYZ::Ptr cloud(new pcl::PointCloudpcl::PointXYZ);
  pcl::fromPCLPointCloud2(mesh.cloud, *cloud);
  pcl::VoxelGridpcl::PointXYZ voxel_grid;
  voxel_grid.setInputCloud(cloud);
  voxel_grid.setLeafSize(0.005, 0.005, 0.005);
  voxel_grid.filter(*cloud);

  // Randomly sample the point cloud
  pcl::PointCloudpcl::PointXYZ::Ptr sampled_cloud(new pcl::PointCloudpcl::PointXYZ);
  pcl::RandomSamplepcl::PointXYZ sampler;
  sampler.setInputCloud(cloud);
  sampler.setSample(num_points);
  sampler.filter(*sampled_cloud);

  // Save the point cloud
  pcl::io::savePCDFileBinary(target_file, *sampled_cloud);
  ROS_INFO_STREAM("Saved point cloud to: " << target_file);

  // Convert the point cloud to a ROS message
  pcl::PCLPointCloud2 pcl_cloud;
  pcl::toPCLPointCloud2(*sampled_cloud, pcl_cloud);
  pcl_conversions::fromPCL(pcl_cloud, res.cloud);

  // Set the number of points after sampling
  res.num_points = sampled_cloud->size();

  return true;
}
