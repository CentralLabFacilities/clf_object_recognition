#include <ros/ros.h>
#include <boost/filesystem.hpp>
#include <sensor_msgs/PointCloud2.h>
// #include "your_package_name/convert_mesh_to_pcl.h" // replace with your actual package and header name

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "convert_meshes_to_pcl");
    ros::NodeHandle nh("~");

    // Get parameters
    std::string models_path;
    std::string pcl_models_path;
    int num_points;
    nh.param<std::string>("models_path", models_path, "");
    nh.param<std::string>("pcl_models_path", pcl_models_path, "");
    nh.param<int>("num_points", num_points, 10000);

    // Find all .dae files in the models directory
    std::vector<std::string> dae_files;
    boost::filesystem::recursive_directory_iterator end_itr;
    for (boost::filesystem::recursive_directory_iterator itr(models_path); itr != end_itr; ++itr) {
        if (boost::filesystem::is_regular_file(itr->path()) && itr->path().extension() == ".dae") {
            dae_files.push_back(itr->path().string());
        }
    }

    ROS_INFO_STREAM("Found " << dae_files.size() << " .dae files");

    // Loop over each .dae file and convert it to a point cloud
    for (const std::string& dae_file : dae_files) {
        // Extract the class name from the file path
        std::string class_name = boost::filesystem::path(dae_file).parent_path().filename().string();

        // Construct the target file path
        std::string target_file = pcl_models_path + "/" + class_name + ".pcd";

        // Convert the mesh to a point cloud
        sensor_msgs::PointCloud2::Request req;
        sensor_msgs::PointCloud2::Response res;
        req.source_file = dae_file;
        req.target_file = target_file;
        req.num_points = num_points;
        if (convertMeshToPclMsg(req, res)) {
            ROS_INFO_STREAM("Converted " << dae_file << " to " << target_file);
        } else {
            ROS_ERROR_STREAM("Failed to convert " << dae_file);
        }
    }

    return 0;
}
