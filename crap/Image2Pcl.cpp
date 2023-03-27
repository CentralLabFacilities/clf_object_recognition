#include <ros/ros.h>
#include <ros/node_handle.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <tf2_ros/transform_listener.h>

#include <vision_msgs/Detection2D.h>
#include "clf_object_recognition_msgs/Detect2D.h"
#include "clf_object_recognition_msgs/Detect3D.h"
#include "clf_object_recognition_msgs/Detect2DImage.h"

#include "clf_object_recognition_msgs/Img2RawPointCloudMsg.h"
#include "clf_object_recognition_msgs/LoadPointCloudMsg.h"
#include "clf_object_recognition_msgs/RegistratedPclMsg.h"
#include "clf_object_recognition_msgs/PclToDetectionMsg.h"

/**

\class Image2Pcl
\brief Converts RGB and depth images to point clouds using 2D object detection information
This class subscribes to RGB and depth images and camera information, synchronizes them, and then
uses 2D object detection information to convert the cropped RGB-D image into a point cloud. The point
cloud is then published on a topic and optionally processed with a voxel grid filter and statistical
outlier removal filter before being published again.
/
class Image2Pcl {
public:
/*
\brief Constructor for Image2Pcl class
\param detect_2d_topic the topic to listen for 2D object detections on
\param publish_raw_pcl whether or not to publish the raw point cloud

*/

class Image2Pcl
{
public:
  Image2Pcl(std::string detect_2d_topic, bool publish_raw_pcl, bool publish_clean_pcl, bool publish_detections)
    : publish_raw_pcl_(publish_raw_pcl)
    , publish_clean_pcl_(publish_clean_pcl)
    , publish_detections_(publish_detections)
    ,

    srv_detect_(nh_.serviceClient<clf_object_recognition_msgs::Detect2DImage>(detect_2d_topic))
    , srv_img_2_raw_pcl_(nh_.serviceClient<clf_object_recognition_msgs::Img2RawPointCloudMsg>("img_2_raw_pcl"))
    , srv_load_pcl_(nh_.serviceClient<clf_object_recognition_msgs::LoadPointCloudMsg>("load_point_cloud"))
    , srv_registrated_pcl_(nh_.serviceClient<clf_object_recognition_msgs::RegistratedPclMsg>("pcl_registration"))
    , srv_detect_3d_(nh_.serviceClient<clf_object_recognition_msgs::PclToDetectionMsg>("pcl_2_detection"))
    ,

    service_(nh_.advertiseService("pcl_registration", &Image2Pcl::service_callback_pcl_pub, this))
  {
    raw_pcl_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("/raw_object_pcl", 1);
    pcl_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("/object_pcl", 1);
    detection_pub_ = nh_.advertise<clf_object_recognition_msgs::Detect3D>("/object_detections", 1);

    image_sub_.subscribe(nh_, "/xtion/rgb/image_raw", 1);
    depth_sub_.subscribe(nh_, "/xtion/depth_registered/image_raw", 1);
    info_sub_.subscribe(nh_, "/xtion/depth_registered/camera_info", 1);

    ts_ = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>(
        image_sub_, depth_sub_, info_sub_, 10);
    ts_->registerCallback(boost::bind(&Image2Pcl::callback, this, _1, _2, _3));
  }

private:
  ros::NodeHandle nh_;    ///< Node handle for this class
  bool publish_raw_pcl_;  ///< Whether or not to publish the raw point cloud
  bool publish_clean_pcl_;
  bool publish_detections_;
  ros::ServiceClient srv_detect_;           ///< Service client for 2D object detection
  ros::ServiceClient srv_img_2_raw_pcl_;    ///< Service client for PCL detection
  ros::ServiceClient srv_load_pcl_;         ///< Service client for loading a pcl by path
  ros::ServiceClient srv_registrated_pcl_;  ///< Service client for registering a pcl
  ros::ServiceClient srv_detect_3d_;        ///< Service client for 3D object detection

  ros::ServiceServer service_;                                     ///< Service server for publishing point clouds
  ros::Publisher raw_pcl_pub_;                                     ///< Publisher for the raw point cloud
  ros::Publisher pcl_pub_;                                         ///< Publisher for the registered point cloud
  ros::Publisher detection_pub_;                                   ///< Publisher for the detection3D
  message_filters::Subscriber<sensor_msgs::Image> image_sub_;      ///< Subscriber for RGB images
  message_filters::Subscriber<sensor_msgs::Image> depth_sub_;      ///< Subscriber for depth images
  message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_;  ///< Subscriber for camera information
  message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>*
      ts_;                                         ///< Time synchronizer for RGB and depth images
  sensor_msgs::Image::ConstPtr image_;             ///< Pointer to RGB image
  sensor_msgs::Image::ConstPtr depth_;             ///< Pointer to depth image
  sensor_msgs::CameraInfo::ConstPtr camera_info_;  ///< Pointer to camera information message

  /**
   * \brief Callback function for RGB, depth, and camera info subscribers
   * \param image RGB image
   * \param depth depth image
   * \param camera_info camera information message
   */
  void callback(const sensor_msgs::Image::ConstPtr& image, const sensor_msgs::Image::ConstPtr& depth,
                const sensor_msgs::CameraInfo::ConstPtr& camera_info)
  {
    image_ = image;
    depth_ = depth;
    camera_info_ = camera_info;
  }

  /**
   * \brief Callback function for point cloud service
   * \param req service request message
   * \param res service response message
   * \return true if successful, false otherwise
   */
  bool service_callback_pcl_pub(clf_object_recognition_msgs::Detect3D::Request& req,
                                clf_object_recognition_msgs::Detect3D::Response& res)
  {
    // Get image, depth, and camera info messages
    sensor_msgs::Image img = *image_;
    sensor_msgs::Image depth_img = *depth_;
    sensor_msgs::CameraInfo info_msg = *camera_info_;

    // Call 2D object detection service
    clf_object_recognition_msgs::Detect2DImage srv_detect;
    srv_detect.request.image = img;

    clf_object_recognition_msgs::Detect2DImageResponse detections_resp;
    if (srv_detect_.call(srv_detect))
    {
      detections_resp = srv_detect.response;
    }
    else
    {
      ROS_ERROR("Failed to call 2D object detection service");
      return false;
    }

    bool success = true;

    // Iterate through the detections
    // for (auto detection : detections) {
    for (vision_msgs::Detection2D detection : detections_resp.detections)
    {
      // Extract detection parameters
      // vision_msgs::ObjectHypothesisWithPose[] results = detection.results;
      const auto& results = detection.results;  // Assuming results is a vector or other container type
      int class_id = results[0].id;
      // std::string class_name = detection.class_name;
      float certainty = results[0].score;
      // int xmin = detection.bbox.xmin;
      // int ymin = detection.bbox.ymin;
      // int xmax = detection.bbox.xmax;
      // int ymax = detection.bbox.ymax;

      // Call img_2_raw_point_cloud_provider
      // img_2_raw_point_cloud_provider::Request req;
      // img_2_raw_point_cloud_provider::Response resp;
      clf_object_recognition_msgs::Img2RawPointCloudMsg::Request req;
      clf_object_recognition_msgs::Img2RawPointCloudMsg::Response resp;
      req.camera_link = "camera_link";
      req.fixed_frame = "world";
      req.image = img;
      req.depth_image = depth_img;
      req.camera_info = info_msg;

      req.bbox = detection.bbox;  // 2D bounding box coordinates
      req.class_id = class_id;    // Object class_id
      req.certainty = certainty;  // Object detection score
      // req.xmin = xmin;
      // req.ymin = ymin;
      // req.xmax = xmax;
      // req.ymax = ymax;
      // req.class_name = class_name;
      // req.certainty = certainty;
      bool success = srv_img_2_raw_pcl_.call(req, resp);

      if (!success)
      {
        ROS_ERROR("Failed to call img_2_raw_point_cloud_provider service");
        break;
      }

      // Publish the point clouds and Detection3Ds
      if (publish_raw_pcl_)
      {
        // raw_pcl_pub_.publish(*raw_point_cloud);
        raw_pcl_pub_.publish(resp.pointcloud);
      }
      // TODO get class_name from class_id
      std::string class_name = "class_name";

      // Call load_point_cloud_provider
      clf_object_recognition_msgs::LoadPointCloudMsg::Request load_req;
      clf_object_recognition_msgs::LoadPointCloudMsg::Response load_resp;
      load_req.filename = "/path/to/point/cloud/for/" + class_name;
      load_req.frame_id = "world";
      success = srv_load_pcl_.call(load_req, load_resp);

      if (!success)
      {
        ROS_ERROR("Failed to call load_point_cloud_provider service");
        break;
      }

      // Call registrated_pcl_provider
      clf_object_recognition_msgs::RegistratedPclMsg::Request reg_req;
      clf_object_recognition_msgs::RegistratedPclMsg::Response reg_resp;
      reg_req.raw_cloud = resp.pointcloud;
      reg_req.ref_cloud = load_resp.pointcloud;
      success = srv_registrated_pcl_.call(reg_req, reg_resp);

      if (!success)
      {
        ROS_ERROR("Failed to call registrated_pcl_provider service");
        break;
      }

      if (publish_clean_pcl_)
      {
        pcl_pub_.publish(reg_resp.pointcloud);
      }

      // Call pcl_to_detection_provider
      clf_object_recognition_msgs::PclToDetectionMsg::Request det_req;
      clf_object_recognition_msgs::PclToDetectionMsg::Response det_resp;
      det_req.pointcloud = reg_resp.pointcloud;
      success = srv_detect_3d_.call(det_req, det_resp);

      if (!success)
      {
        ROS_ERROR("Failed to call pcl_to_detection_provider service");
        break;
      }

      if (publish_detections_)
      {
        detection_pub_.publish(det_resp.detection);
      }

      // Do something with the results
      // ...

      return true;

    }  // end for detections
  }

  // Image2Pcl::~Image2Pcl() {
  //    delete ts_;
  //}
};
