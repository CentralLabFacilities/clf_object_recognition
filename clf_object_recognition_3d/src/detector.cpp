#include "clf_object_recognition_3d/detector.h"
#include "clf_object_recognition_msgs/Detect2DImage.h"

#include "geometry_msgs/Pose.h"

#include "pcl/point_cloud.h"
#include "pcl/point_types.h"

#include "pcl/common/centroid.h"
#include "pcl/common/eigen.h"
//#include "pcl_ros/io/pcl_conversions.h"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/registration/icp.h>

//#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <pcl/io/vtk_lib_io.h>
//#include <pcl/io/vtk_lib_io.h>

#include "geometric_shapes/mesh_operations.h"
#include <Eigen/Geometry>

#include <ros/console.h>

#include <cv_bridge/cv_bridge.h>
#include <image_geometry/pinhole_camera_model.h>

#include <eigen_conversions/eigen_msg.h>

#include "clf_object_recognition_3d/cloud_sampler.hpp"
#include "clf_object_recognition_3d/cloud_from_image.h"

inline bool validateFloats(double val)
{
  return !(std::isnan(val) || std::isinf(val));
}

Detector::Detector(ros::NodeHandle nh) : sync_(image_sub_, depth_image_sub_, camera_info_sub_, 10)
{
  auto f = [this](auto&& PH1, auto&& PH2) { ReconfigureCallback(PH1, PH2); };
  reconfigure_server.setCallback(f);
  // get configuration first
  ros::spinOnce();

  srv_detect_2d = nh.serviceClient<clf_object_recognition_msgs::Detect2DImage>("/yolox/recognize_from_image");
  srv_detect_3d = nh.advertiseService("simple_detections", &Detector::ServiceDetect3D, this);

  pub_detections_3d = nh.advertise<vision_msgs::Detection3DArray>("last_detection", 1);
  pub_marker = nh.advertise<visualization_msgs::MarkerArray>("objects", 1);

  // vision_msgs::Detection3DArray

  // subscribe to camera topics
  image_sub_.subscribe(nh, config.image_topic, 1);
  depth_image_sub_.subscribe(nh, config.depth_topic, 1);
  camera_info_sub_.subscribe(nh, config.info_topic, 1);

  // sync incoming camera messages
  sync_.registerCallback(boost::bind(&Detector::Callback, this, _1, _2, _3));

  model_provider = std::make_unique<ModelProvider>(nh);
}

void Detector::Callback(const sensor_msgs::ImageConstPtr& image, const sensor_msgs::ImageConstPtr& depth_image,
                        const sensor_msgs::CameraInfoConstPtr& camera_info)
{
  std::lock_guard<std::mutex> lock(mutex_);
  image_ = image;
  depth_image_ = depth_image;
  camera_info_ = camera_info;
}

void Detector::ReconfigureCallback(const clf_object_recognition_cfg::Detect3dConfig& input, uint32_t /*level*/)
{
  ROS_INFO_NAMED("detector", "Reconfigure");
  config = input;
}

bool Detector::ServiceDetect3D(clf_object_recognition_msgs::Detect3D::Request& req,
                               clf_object_recognition_msgs::Detect3D::Response& res)
{
  sensor_msgs::Image img;
  sensor_msgs::Image depth;
  sensor_msgs::CameraInfo info;

  ROS_INFO_STREAM_NAMED("detector", "ServiceDetect3D() called " << req);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (image_ == nullptr) {
      ROS_ERROR_STREAM_NAMED("detector", "ServiceDetect3D() called: No rgb image provided " << req);
      return false;
    } else if (depth_image_ == nullptr) {
      ROS_ERROR_STREAM_NAMED("detector", "ServiceDetect3D() called: No depth image provided" << req);
      return false;
    }
    img = sensor_msgs::Image(*image_.get());
    depth = sensor_msgs::Image(*depth_image_.get());
  }

  ROS_INFO_STREAM_NAMED("detector", "ServiceDetect3D() got images ");

  clf_object_recognition_msgs::Detect2DImage param;
  param.request.image = img;

  auto ok = srv_detect_2d.call(param);
  if (!ok)
  {
    ROS_ERROR_STREAM_NAMED("detector", "cant call detections ");
    return false;
  }

  visualization_msgs::MarkerArray markers;

  int i = 0;

  ROS_DEBUG_STREAM_NAMED("detector", "got detections " << param.response);
  for (auto& detection : param.response.detections)
  {
    ROS_DEBUG_STREAM_NAMED("detector", "detection");
    vision_msgs::Detection3D d3d;
    // generate point cloud from incoming depth image for detection bounding box
    pointcloud_type::Ptr cloud_from_depth_image = cloud::fromDepthArea(detection.bbox, depth, *camera_info_);
    //pointcloud_type::Ptr cloud_from_depth_image = cloud::oldFromDepth( depth, detection.bbox, camera_info_);
    // pointcloud_type* cloud_from_mesh = createPointCloudFromMesh(mesh_name);

    Eigen::Vector4d cloud_from_depth_image_centroid = Eigen::Vector4d::Random();
    auto centroid_size = pcl::compute3DCentroid(*cloud_from_depth_image, cloud_from_depth_image_centroid);

    geometry_msgs::Pose center;
    center.orientation.w = 1;
    center.position.x = cloud_from_depth_image_centroid[0];
    center.position.y = cloud_from_depth_image_centroid[1];
    center.position.z = cloud_from_depth_image_centroid[2];

    if (center.position.x != center.position.x)
      center.position.x = 0.1;
    if (center.position.y != center.position.y)
      center.position.y = 0.1;
    if (center.position.z != center.position.z)
      center.position.z = 0.1;

    ROS_DEBUG_STREAM_NAMED("detector", "      center at " << center.position.x << ", " << center.position.y << ", " << center.position.z);

    d3d.header = detection.header;
    d3d.bbox.center = center;
    d3d.bbox.size.x = 0.1;
    d3d.bbox.size.y = 0.1;
    d3d.bbox.size.z = 0.1;

    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(*cloud_from_depth_image, pcl_msg);
    d3d.source_cloud = pcl_msg;
    for (auto hypo : detection.results)
    {
      ROS_DEBUG_STREAM_NAMED("detector", "  - hypo " << hypo.id);
      visualization_msgs::Marker marker;
      marker.type = visualization_msgs::Marker::MESH_RESOURCE;
      marker.id = i++;
      marker.header = img.header;
      marker.scale.x = 1;
      marker.scale.y = 1;
      marker.scale.z = 1;
      marker.color.a = 0.5;

      vision_msgs::ObjectHypothesisWithPose hyp;
      hyp = hypo;
      hyp.pose.pose.orientation.w = 1;

      auto model = model_provider->IDtoModel(hypo.id);
      auto path = model_provider->GetModelPath(model);
      if (path == "")
      {
        // default if not found (e.g. ecwm not running)
        path = "file:///vol/tiago/noetic/nightly/share/ecwm_data/models/object/ycb/cracker_box/meshes/textured.dae";
      }
      marker.mesh_resource = path;

      // create polygon mesh from .dae ressource
      ROS_DEBUG_STREAM_NAMED("detector", "      load mesh");
      mesh_type::Ptr reference_mesh = colladaToPolygonMesh(path);
      ROS_DEBUG_STREAM_NAMED("detector", "      sample mesh");
      auto sampled = sample_cloud(reference_mesh);

      Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
      transformation_matrix(0, 3) = cloud_from_depth_image_centroid[0];
      transformation_matrix(1, 3) = cloud_from_depth_image_centroid[1];
      transformation_matrix(2, 3) = cloud_from_depth_image_centroid[2];
      pcl::transformPointCloud(*sampled, *sampled, transformation_matrix);

      pcl::IterativeClosestPoint<point_type, point_type> icp;
      icp.setInputSource(sampled);
      icp.setInputTarget(cloud_from_depth_image);
      pointcloud_type final_point_cloud;
      ROS_DEBUG_STREAM_NAMED("detector", "      icp");
      icp.align(final_point_cloud);
      Eigen::Matrix4f icp_transform = icp.getFinalTransformation();
      Eigen::Matrix4f transformation_inverse = transformation_matrix.cast<float>().inverse();
      Eigen::Matrix4f final_transformation = transformation_inverse * icp_transform;
      Eigen::Affine3f affine(final_transformation);
      
      geometry_msgs::Transform tf_msg;
      tf::transformEigenToMsg(affine.cast<double>(), tf_msg);

      ROS_DEBUG_STREAM_NAMED("detector", "      center at " << tf_msg.translation.x << ", " << tf_msg.translation.y << ", " << tf_msg.translation.z);
      hyp.pose.pose.orientation = tf_msg.rotation;
      hyp.pose.pose.position.x = tf_msg.translation.x;
      hyp.pose.pose.position.y = tf_msg.translation.y;
      hyp.pose.pose.position.z = tf_msg.translation.z;
      marker.pose = hyp.pose.pose;

      ROS_INFO_STREAM_NAMED("detector", "      has converged: " << icp.hasConverged());
      // FIXME: icp score calculation not working
      // auto d = icp.getFitnessScore();
      // ROS_INFO_STREAM_NAMED("detector", "has converged: " << icp.hasConverged() << " score: " << d);
      // ROS_INFO_STREAM_NAMED("detector", "final transformation: " << icp.getFinalTransformation());
      markers.markers.push_back(marker);
      d3d.results.push_back(hyp);
    }

    res.detections.push_back(d3d);
  }

  if (config.publish_marker)
  {
    pub_marker.publish(markers);
  }

  if (config.publish_detections)
  {
    ROS_DEBUG_STREAM_NAMED("detector", "publishing detections ");
    vision_msgs::Detection3DArray a;
    for (auto b : res.detections)
    {
      a.header = b.header;
      a.detections.push_back(b);
    }
    pub_detections_3d.publish(a);
  }

  return true;
}

// pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr Detector::meshSamplingUniform(const vtkSmartPointer<vtkPolyData>&
// polydata, int samples, bool calcNormal) {
//   pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());

//    return cloud;
//}

mesh_type::Ptr Detector::colladaToPolygonMesh(const std::string& ressource_path)
{
  // TODO: Check if header is needed
  shapes::Mesh* reference_mesh = shapes::createMeshFromResource(ressource_path);
  mesh_type polygon_mesh{};

  pointcloud_type::Ptr cloud(new pointcloud_type());
  cloud->points.resize(reference_mesh->vertex_count);

  pointcloud_type::iterator pt_iter = cloud->begin();
  for (int offset = 0; offset < reference_mesh->vertex_count; offset++, ++pt_iter)
  {
    point_type& pt = *pt_iter;
    pt.x = reference_mesh->vertices[offset];
    pt.y = reference_mesh->vertices[offset + 1];
    pt.z = reference_mesh->vertices[offset + 2];
  }

  sensor_msgs::PointCloud2 pcl_msg;
  pcl::toROSMsg(*cloud.get(), pcl_msg);

  std::vector<pcl::Vertices> vertex_indices;
  for (int offset = 0; offset < reference_mesh->triangle_count; offset++)
  {
    pcl::Vertices polygon_vertices;
    polygon_vertices.vertices.push_back(reference_mesh->triangles[offset]);
    polygon_vertices.vertices.push_back(reference_mesh->triangles[offset + 1]);
    polygon_vertices.vertices.push_back(reference_mesh->triangles[offset + 2]);
    vertex_indices.push_back(polygon_vertices);
  }

  // polygon_mesh.header = header;
  pcl_conversions::toPCL(pcl_msg, polygon_mesh.cloud);
  polygon_mesh.polygons = vertex_indices;

  return std::make_shared<pcl::PolygonMesh>(polygon_mesh);
}
