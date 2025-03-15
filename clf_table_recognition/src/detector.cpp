#include "clf_table_recognition/detector.h"
#include <vector>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <tf2_eigen/tf2_eigen.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

Detector::Detector(ros::NodeHandle nh)
  :  nh_(nh), tf_listener(tf_buffer)
{

  srv_table_ = nh.advertiseService("get_table_height", &Detector::ServiceDetectTable, this);

  pub_cloud_table_ = nh.advertise<sensor_msgs::PointCloud2>("cloud", 1);

}

bool Detector::ServiceDetectTable(clf_object_recognition_msgs::GetFloat::Request& req,
    clf_object_recognition_msgs::GetFloat::Response& res)
{
    ROS_INFO_STREAM_NAMED("detector", "called detect table:" << req);

    ROS_INFO_NAMED("detector", "waiting for cloud");
    auto msg = ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/camera/depth/points",nh_, ros::Duration(5));
    
    if (msg == nullptr)
    {
        ROS_WARN_STREAM_NAMED("detector", "failed to get point cloud");
        return false;
    }
    ROS_INFO_NAMED("detector", "got cloud");

    pcl::PointCloud<point_t>::Ptr cloud(new pcl::PointCloud<point_t>);
    {
        geometry_msgs::TransformStamped transform_stamped;
        try
        {
            // get latest as pcl could be after latest transform publisher update
            transform_stamped = tf_buffer.lookupTransform("base_footprint", msg->header.frame_id, ros::Time(0));
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN_STREAM_NAMED("detector", ex.what());
            return false;
        }
        ROS_DEBUG_STREAM_NAMED("detector", "getPointCloud(): successfully got transform");
        const Eigen::Isometry3d e = tf2::transformToEigen(transform_stamped);
        sensor_msgs::PointCloud2 trans_cloud;
        // Seems to work only like this, else this error appears
        // http://eigen.tuxfamily.org/dox-devel/group__TopicUnalignedArrayAssert.html
        const Eigen::Matrix4f transform_matrix = e.matrix().cast<float>();
        ROS_DEBUG_STREAM_NAMED("detector", "Calling transformPointCloud ...");
        // TODO(mvieth) this line takes about 0.75s. Make that faster if possible
        // In pcl 1.9.0, it is much faster through SIMD maybe
        // transform to pcl cloud first, then transform?
        pcl_ros::transformPointCloud(transform_matrix, *msg, trans_cloud);
        trans_cloud.header.frame_id = "base_footprint";
        ROS_DEBUG_STREAM_NAMED("detector", ";;done");
        ROS_DEBUG_STREAM_NAMED("detector", "Calling fromROSMsg ...trans_cloud");
        pcl::moveFromROSMsg(trans_cloud, *cloud);
        ROS_DEBUG_STREAM_NAMED("detector", "..done");
    }


    pcl::PointCloud<point_t>::Ptr cloud_voxel(new pcl::PointCloud<point_t>);
    {
        // Create the filtering object
        pcl::VoxelGrid<point_t> sor;
        sor.setInputCloud (cloud);
        sor.setLeafSize (0.01f, 0.01f, 0.01f);
        sor.filter (*cloud_voxel);
    }

    pcl::PointCloud<point_t>::Ptr cloud_filtered(new pcl::PointCloud<point_t>);
    {
        pcl::CropBox<point_t> cropBoxFilter (true);
        cropBoxFilter.setInputCloud (cloud_voxel);
        Eigen::Vector4f min_pt (-1.0f, -1.0f, 0.3f, 1.0f);
        Eigen::Vector4f max_pt (1.0f, 1.0f, 2.0f, 1.0f);

        // Cropbox slighlty bigger then bounding box of points
        cropBoxFilter.setMin (min_pt);
        cropBoxFilter.setMax (max_pt);

        // Cloud  
        cropBoxFilter.filter (*cloud_filtered);
    }

    res.value = 1.0;
    return true;
    

    

}