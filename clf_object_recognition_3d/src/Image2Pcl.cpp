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

#include "clf_object_recognition_msgs/Detect3D.h"
#include "clf_object_recognition_msgs/Detect2DImage.h"

#include "img_2_raw_point_cloud_provider.h"
#include "load_point_cloud_provider.h"
#include "registrated_pcl_provider.h"
#include "pcl_to_detection_provider.h"


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

class Image2Pcl {
public:
    Image2Pcl(std::string detect_2d_topic, bool publish_raw_pcl, bool publish_clean_pcl, bool publish_detections) :
        publish_raw_pcl_(publish_raw_pcl),
        publish_clean_pcl_(publish_clean_pcl),
        publish_detections_(publish_detections),
        srv_detect_(nh_.serviceClient<clf_object_recognition_msgs::Detect2DImage>(detect_2d_topic)),
        service_(nh_.advertiseService("pcl_registration", &Image2Pcl::service_callback_pcl_pub, this))
    {
        raw_pcl_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("/raw_object_pcl", 1);
        pcl_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("/object_pcl", 1);
        detection_pub_ = nh_.advertise<clf_object_recognition_msgs::Detection3DArray>("/object_detections", 1);
        
        image_sub_.subscribe(nh_, "/xtion/rgb/image_raw", 1);
        depth_sub_.subscribe(nh_, "/xtion/depth_registered/image_raw", 1);
        info_sub_.subscribe(nh_, "/xtion/depth_registered/camera_info", 1);

        ts_ = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>(image_sub_, depth_sub_, info_sub_, 10);
        ts_->registerCallback(boost::bind(&Image2Pcl::callback, this, _1, _2, _3));
    }

private:
    ros::NodeHandle nh_; ///< Node handle for this class
    bool publish_raw_pcl_; ///< Whether or not to publish the raw point cloud
    bool publish_clean_pcl_;
    bool publish_detections_;
    ros::ServiceClient srv_detect_; ///< Service client for 2D object detection
    ros::ServiceServer service_; ///< Service server for publishing point clouds
    ros::Publisher raw_pcl_pub_; ///< Publisher for the raw point cloud
    ros::Publisher pcl_pub_; ///< Publisher for the registered point cloud
    ros::Publisher detection_pub_; ///< Publisher for the detection3D
    message_filters::Subscriber<sensor_msgs::Image> image_sub_; ///< Subscriber for RGB images
    message_filters::Subscriber<sensor_msgs::Image> depth_sub_; ///< Subscriber for depth images
    message_filters::Subscriber<sensor_msgs::CameraInfo> info_sub_; ///< Subscriber for camera information
    message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>* ts_; ///< Time synchronizer for RGB and depth images
    sensor_msgs::Image::ConstPtr image_; ///< Pointer to RGB image
    sensor_msgs::Image::ConstPtr depth_; ///< Pointer to depth image
    sensor_msgs::CameraInfo::ConstPtr camera_info_; ///< Pointer to camera information message

    /**
     * \brief Callback function for RGB, depth, and camera info subscribers
     * \param image RGB image
     * \param depth depth image
     * \param camera_info camera information message
     */
    void callback(const sensor_msgs::Image::ConstPtr& image, const sensor_msgs::Image::ConstPtr& depth, const sensor_msgs::CameraInfo::ConstPtr& camera_info) {
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
    bool service_callback_pcl_pub(clf_object_recognition_msgs::Detect3D::Request& req, clf_object_recognition_msgs::Detect3D::Response& res) {
        // Get image, depth, and camera info messages
        sensor_msgs::Image img = *image_;
        sensor_msgs::Image depth_img = *depth_;
        sensor_msgs::CameraInfo info_msg = *camera_info_;

        // Call 2D object detection service
        clf_object_recognition_msgs::Detect2DImage srv_detect;
        srv_detect.request.image = img;

        clf_object_recognition_msgs::Detect2DImageResponse detections;
        if (srv_detect_.call(srv_detect)) {
            detections = srv_detect.response;
        } else {
            ROS_ERROR("Failed to call 2D object detection service");
            return false;
        }

        bool success = true;

        // Iterate through the detections
        for (auto detection : detections) {
            // Extract detection parameters
            std::string class_name = detection.class_name;
            float certainty = detection.certainty;
            int xmin = detection.bbox.xmin;
            int ymin = detection.bbox.ymin;
            int xmax = detection.bbox.xmax;
            int ymax = detection.bbox.ymax;

            // Call img_2_raw_point_cloud_provider
            img_2_raw_point_cloud_provider::Request req;
            img_2_raw_point_cloud_provider::Response resp;
            req.camera_link = "camera_link";
            req.fixed_frame = "world";
            req.image = img;
            req.depth_image = depth_img;
            req.camera_info = info_msg;
            req.xmin = xmin;
            req.ymin = ymin;
            req.xmax = xmax;
            req.ymax = ymax;
            req.class_name = class_name;
            req.certainty = certainty;
            bool success = img_2_raw_point_cloud_provider_client.call(req, resp);

            if (!success) {
                ROS_ERROR("Failed to call img_2_raw_point_cloud_provider service");
                break;
            }

            // Publish the point clouds and Detection3Ds
            if (publish_raw_pcl_) {
                raw_pcl_pub_.publish(*raw_point_cloud);
            }

            // Call load_point_cloud_provider
            load_point_cloud_provider::Request load_req;
            load_point_cloud_provider::Response load_resp;
            load_req.filename = "/path/to/point/cloud/for/" + class_name;
            load_req.frame_id = "world";
            success = load_point_cloud_provider_client.call(load_req, load_resp);

            if (!success) {
                ROS_ERROR("Failed to call load_point_cloud_provider service");
                break;
            }

            // Call registrated_pcl_provider
            registrated_pcl_provider::Request reg_req;
            registrated_pcl_provider::Response reg_resp;
            reg_req.raw_cloud = resp.pointcloud;
            reg_req.ref_cloud = load_resp.cloud;
            success = registrated_pcl_provider_client.call(reg_req, reg_resp);

            if (!success) {
                ROS_ERROR("Failed to call registrated_pcl_provider service");
                break;
            }

            if (publish_clean_pcl_) {
                pcl_pub_.publish(*registered_point_cloud);
            }

            // Call pcl_to_detection_provider
            pcl_to_detection_provider::Request pcl_req;
            pcl_to_detection_provider::Response pcl_resp;
            pcl_req.cloud = reg_resp.registered_cloud;
            success = pcl_to_detection_provider_client.call(pcl_req, pcl_resp);

            if (!success) {
                ROS_ERROR("Failed to call pcl_to_detection_provider service");
                break;
            }

            if (publish_detections_) {
                detection_pub_.publish(detections_3d);
            }

            // Do something with the results
            // ...

        } // end for detections

    }

    //Image2Pcl::~Image2Pcl() {
    //    delete ts_;
    //}

return true;
};
