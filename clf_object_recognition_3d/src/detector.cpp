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

#include <eigen_conversions/eigen_msg.h>

#include "clf_object_recognition_3d/cloud_sampler.hpp"

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

  {
    std::lock_guard<std::mutex> lock(mutex_);
    img = sensor_msgs::Image(*image_.get());
    depth = sensor_msgs::Image(*depth_image_.get());
  }

  ROS_INFO_STREAM_NAMED("detector", "ServiceDetect3D() called " << req);

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

  ROS_INFO_STREAM_NAMED("detector", "got detections " << param.response);
  for (auto& detection : param.response.detections)
  {
    vision_msgs::Detection3D d3d;
    // generate point cloud from incoming depth image for detection bounding box
    pointcloud_type::Ptr cloud_from_depth_image = createPointCloudFromDepthImage(depth, detection.bbox, camera_info_);
    // pointcloud_type* cloud_from_mesh = createPointCloudFromMesh(mesh_name);

    Eigen::Vector4d cloud_from_depth_image_centroid = Eigen::Vector4d::Random();
    auto centroid_size = pcl::compute3DCentroid(*cloud_from_depth_image, cloud_from_depth_image_centroid);

    geometry_msgs::Pose centroid_pose_msg;
    centroid_pose_msg.position.x = cloud_from_depth_image_centroid[0];
    centroid_pose_msg.position.y = cloud_from_depth_image_centroid[1];
    centroid_pose_msg.position.z = cloud_from_depth_image_centroid[2];

    d3d.header = detection.header;
    d3d.bbox.center.orientation.w = 1;
    d3d.bbox.center.position.x = cloud_from_depth_image_centroid.x();
    d3d.bbox.center.position.y = cloud_from_depth_image_centroid.y();
    d3d.bbox.center.position.z = cloud_from_depth_image_centroid.z();
    d3d.bbox.size.x = 0.1;
    d3d.bbox.size.y = 0.1;
    d3d.bbox.size.z = 0.1;

    sensor_msgs::PointCloud2 pcl_msg;
    pcl::toROSMsg(*cloud_from_depth_image, pcl_msg);
    d3d.source_cloud = pcl_msg;
    for (auto hypo : detection.results)
    {
      visualization_msgs::Marker marker;
      marker.type = visualization_msgs::Marker::MESH_RESOURCE;
      marker.id = i++;
      marker.header = img.header;
      marker.scale.x = 1;
      marker.scale.y = 1;
      marker.scale.z = 1;
      marker.color.a = 1;

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
      mesh_type::Ptr reference_mesh = colladaToPolygonMesh(path);
      auto sampled = sample_cloud(reference_mesh);

      // uniform mesh sampling
      // https://github.com/PointCloudLibrary/pcl/blob/master/tools/mesh_sampling.cpp

      pcl::IterativeClosestPoint<point_type, point_type> icp;
      icp.setInputSource(sampled);
      icp.setInputTarget(cloud_from_depth_image);
      pointcloud_type final_point_cloud;
      icp.align(final_point_cloud);
      auto transform = icp.getFinalTransformation();
      Eigen::Affine3f affine(transform);
      geometry_msgs::Transform tf_msg;
      tf::transformEigenToMsg(affine.cast<double>(), tf_msg);

      hyp.pose.pose.orientation = tf_msg.rotation;
      hyp.pose.pose.position.x = tf_msg.translation.x;
      hyp.pose.pose.position.y = tf_msg.translation.y;
      hyp.pose.pose.position.z = tf_msg.translation.z;
      marker.pose = hyp.pose.pose;
      // TODO: remove later

      ROS_INFO_STREAM_NAMED("detector", "has converged: " << icp.hasConverged());
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
    ROS_INFO_STREAM_NAMED("detector", "publishing detections ");
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

pointcloud_type::Ptr Detector::createPointCloudFromMesh(const mesh_type::Ptr& mesh)
{
  pointcloud_type::Ptr cloud(new pointcloud_type());
  // generate point cloud
  return cloud;
}

pointcloud_type::Ptr Detector::createPointCloudFromDepthImage(const sensor_msgs::Image& depth_msg,
                                                              const vision_msgs::BoundingBox2D& bbox,
                                                              const sensor_msgs::CameraInfoConstPtr& cam_info)
{
  pointcloud_type::Ptr cloud(new pointcloud_type());

  cloud->header.stamp = ros::Time(depth_msg.header.stamp).toSec();
  cloud->header.frame_id = depth_msg.header.frame_id;
  ROS_INFO_STREAM_NAMED("Detector ", "frame_id: " << depth_msg.header.frame_id);
  // single point of view, 2d rasterized
  cloud->is_dense = true;

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

  pointcloud_type::iterator pt_iter = cloud->begin();
  for (int v = (int)(bbox.center.y - bbox.size_y / 2); v < (int)(bbox.center.y - bbox.size_y / 2 + bbox.size_y); ++v)
  {
    for (int u = (int)(bbox.center.x - bbox.size_x / 2); u < (int)(bbox.center.x - bbox.size_x / 2 + bbox.size_x);
         ++u, ++pt_iter)
    {
      point_type& pt = *pt_iter;
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
