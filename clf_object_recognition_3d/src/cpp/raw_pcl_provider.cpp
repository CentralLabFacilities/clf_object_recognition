#include <ros/ros.h>
#include <ros/node_handle.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <clf_object_recognition_msgs/Detect3D.h>
#include <clf_object_recognition_msgs/Detect2DImage.h>


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
    Image2Pcl(std::string detect_2d_topic, bool publish_raw_pcl) :
        publish_raw_pcl_(publish_raw_pcl),
        srv_detect_(nh_.serviceClient<clf_object_recognition_msgs::Detect2DImage>(detect_2d_topic)),
        service_(nh_.advertiseService("pcl_registration", &Image2Pcl::service_callback_pcl_pub, this))
    {
        raw_pcl_pub_ = nh_.advertise<pcl::PointCloud<pcl::PointXYZ>>("/raw_object_pcl", 1);
        
        image_sub_.subscribe(nh_, "/xtion/rgb/image_raw", 1);
        depth_sub_.subscribe(nh_, "/xtion/depth_registered/image_raw", 1);
        info_sub_.subscribe(nh_, "/xtion/depth_registered/camera_info", 1);

        ts_ = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo>(image_sub_, depth_sub_, info_sub_, 10);
        ts_->registerCallback(boost::bind(&Image2Pcl::callback, this, _1, _2, _3));
    }

private:
    ros::NodeHandle nh_; ///< Node handle for this class
    bool publish_raw_pcl_; ///< Whether or not to publish the raw point cloud
    ros::ServiceClient srv_detect_; ///< Service client for 2D object detection
    ros::ServiceServer service_; ///< Service server for publishing point clouds
    ros::Publisher raw_pcl_pub_; ///< Publisher for the raw point cloud
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
        clf_object_recognition_msgs::Detect2DImage srv;
        srv.request.image = img;

        clf_object_recognition_msgs::Detect2DImageResponse detections;
        if (srv_detect_.call(srv)) {
            detections = srv.response;
        } else {
            ROS_ERROR("Failed to call object detection service");
            return false;
        }

        // Loop through detections
        for (const auto& d2d : detections.detections) {
            // Extract class name and certainty score
            std::string class_name = d2d.results[0].class_name;
            float certainty = d2d.results[0].score;

            // Extract bounding box coordinates
            float xc = d2d.bbox.center.x;
            float yc = d2d.bbox.center.y;
            float width = d2d.bbox.size_x;
            float height = d2d.bbox.size_y;
            int xmin = static_cast<int>(xc - width / 2);
            int ymin = static_cast<int>(yc - height / 2);
            int xmax = static_cast<int>(xc + width / 2);
            int ymax = static_cast<int>(yc + height / 2);

            // Crop RGB and depth images
            cv::Mat img_mat = cv_bridge::toCvCopy(img)->image;
            cv::Mat rgb_crop = img_mat(cv::Range(ymin, ymax), cv::Range(xmin, xmax));
            cv::Mat depth_mat = cv_bridge::toCvCopy(depth_img)->image;
            cv::Mat depth_crop = depth_mat(cv::Range(ymin, ymax), cv::Range(xmin, xmax));

            // Convert depth image to point cloud
            pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr pcl_filtered(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointXYZ point;
            point.z = 0;
            for (int r = 0; r < depth_crop.rows; r++) {
                for (int c = 0; c < depth_crop.cols; c++) {
                    ushort depth_val = depth_crop.at<ushort>(r, c);
                    if (depth_val != 0) {
                        cv::Point3f pt = depthTo3D(cv::Point2i(c + xmin, r + ymin), depth_val, info_msg);
                        point.x = pt.x;
                        point.y = pt.y;
                        pcl_cloud->points.push_back(point);
                    }
                }
            }
            // pcl_cloud->width = pcl_cloud->points.size();
            pcl_cloud->width = pcl_cloud->points.size() / pcl_cloud->height;
            pcl_cloud->height = 1;

            // Apply voxel grid filter
            pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
            voxel_grid.setInputCloud(pcl_cloud);
            voxel_grid.setLeafSize(0.01f, 0.01f, 0.01f);
            voxel_grid.filter(*pcl_filtered);

            // Convert filtered point cloud to PCL XYZRGB format
            pcl::PointCloudpcl::PointXYZRGB::Ptr pcl_colored_cloud(new pcl::PointCloudpcl::PointXYZRGB);
            pcl::copyPointCloud(*pcl_filtered, *pcl_colored_cloud);

            // Apply statistical outlier removal filter
            pcl::StatisticalOutlierRemovalpcl::PointXYZRGB sor;
            sor.setInputCloud(pcl_colored_cloud);
            sor.setMeanK(50);
            sor.setStddevMulThresh(1.0);
            sor.filter(*pcl_colored_cloud);

            // Convert colored point cloud back to XYZ format
            pcl::PointCloudpcl::PointXYZ::Ptr pcl_outliers_removed_cloud(new pcl::PointCloudpcl::PointXYZ);
            pcl::copyPointCloud(*pcl_colored_cloud, *pcl_outliers_removed_cloud);

            // Cluster remaining points in the point cloud
            std::vector<pcl::PointCloudpcl::PointXYZ::Ptr> pcl_clusters;
            pcl::EuclideanClusterExtractionpcl::PointXYZ ec;
            ec.setClusterTolerance(0.02);
            ec.setMinClusterSize(50);
            ec.setMaxClusterSize(10000);
            ec.setInputCloud(pcl_outliers_removed_cloud);
            pcl::search::KdTreepcl::PointXYZ::Ptr tree(new pcl::search::KdTreepcl::PointXYZ);
            ec.setSearchMethod(tree);
            ec.extract(pcl_clusters);

            // Loop through clusters
            for (const auto& cluster : pcl_clusters) {
                // Create a new instance of the 3D detection message
                clf_object_recognition_msgs::Detection3D detection;

                // Set class name and certainty score
                detection.result.class_name = class_name;
                detection.result.score = certainty;

                // Convert cluster to a bounding box
                pcl::PointXYZ min_point, max_point;
                pcl::getMinMax3D(*cluster, min_point, max_point);
                detection.bbox.center.x = (min_point.x + max_point.x) / 2;
                detection.bbox.center.y = (min_point.y + max_point.y) / 2;
                detection.bbox.center.z = (min_point.z + max_point.z) / 2;
                detection.bbox.size_x = max_point.x - min_point.x;
                detection.bbox.size_y = max_point.y - min_point.y;
                detection.bbox.size_z = max_point.z - min_point.z;

                // Add detection to response message
                res.detections.push_back(detection);
            }

    Image2Pcl::~Image2Pcl() {
        delete ts_;
    }

return true;
}
