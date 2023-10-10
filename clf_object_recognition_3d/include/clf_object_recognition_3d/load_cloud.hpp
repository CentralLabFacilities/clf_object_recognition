#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/console/print.h>
#include <pcl/console/parse.h>
#include <boost/algorithm/string/split.hpp>  // for split
#include <filesystem>

#include "geometric_shapes/mesh_operations.h"

namespace cloud
{

pcl::PolygonMesh::Ptr loadPLY(std::filesystem::path path)
{
  pcl::PolygonMesh mesh;
  auto size = pcl::io::loadPLYFile(path, mesh);
  ROS_DEBUG_STREAM_NAMED("cloud", "ply file loaded ");
  return std::make_shared<mesh_type>(mesh);
}

// from https://github.com/PointCloudLibrary/pcl/blob/master/tools/xyz2pcd.cpp
bool loadXYZ(std::filesystem::path path, pcl::PointCloud<pcl::PointXYZ>& cloud)
{
  std::ifstream fs;
  fs.open(path, std::ios::binary);
  if (!fs.is_open() || fs.fail())
  {
    PCL_ERROR("Could not open file '%s'! Error : %s\n", path.c_str(), strerror(errno));
    fs.close();
    return (false);
  }

  std::string line;
  std::vector<std::string> st;

  while (!fs.eof())
  {
    std::getline(fs, line);
    // Ignore empty lines
    if (line.empty())
      continue;

    // Tokenize the line
    boost::trim(line);
    boost::split(st, line, boost::is_any_of("\t\r "), boost::token_compress_on);

    if (st.size() < 3)
      continue;

    cloud.push_back(pcl::PointXYZ(static_cast<float>(atof(st[0].c_str())), static_cast<float>(atof(st[1].c_str())),
                                  static_cast<float>(atof(st[2].c_str()))));
  }
  fs.close();

  cloud.width = cloud.size();
  cloud.height = 1;
  cloud.is_dense = true;
  return true;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr loadPointcloud(const std::string& ressource_path)
{
  namespace fs = std::filesystem;
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

  ROS_DEBUG_STREAM_NAMED("cloud", "loadPointcloud for model path: " << ressource_path);

  std::string real_path = ressource_path;
  std::string prefix("file://");
  real_path.replace(0, prefix.length(), "");

  fs::path path = { real_path };

  if (!fs::exists(path))
  {
    ROS_ERROR_STREAM_NAMED("cloud", "file does not exist: " << path);
    throw std::runtime_error("file does not exist: " + ressource_path);
  }

  bool loaded = false;
  if (fs::exists(path.replace_extension("xyz")))
  {
    ROS_DEBUG_STREAM_NAMED("cloud", "loading xyz: " << path);
    if(loadXYZ(path, *cloud)) return cloud;
  }

  if (fs::exists(path.replace_extension("ply")))
  {
    ROS_DEBUG_STREAM_NAMED("cloud", "loading ply: " << path);
    //auto ply = loadPLY(path);
    //cloud = sample_cloud(ply);
    //return cloud;
  }

  // TODO sample from collada 

  throw std::runtime_error("failed to load pointcloud for: " + ressource_path);
}

}  // namespace cloud
