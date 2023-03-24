#include "clf_object_recognition_rviz/point_cloud_visual.h"

#include "sensor_msgs/PointCloud2.h"
#include "rviz/default_plugin/point_cloud_transformers.h"

#include <boost/make_shared.hpp>

namespace objrec
{
namespace viz
{

PointCloudVisual::PointCloudVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node) {

  scene_manager_ = scene_manager;
  object_node_ = parent_node->createChildSceneNode();

  cloud_.reset(new rviz::PointCloud());
  cloud_->setRenderMode(rviz::PointCloud::RenderMode::RM_POINTS);

  object_node_->attachObject(cloud_.get());

}
PointCloudVisual::~PointCloudVisual() {

}

void PointCloudVisual::setMessage(const sensor_msgs::PointCloud2& msg) {
  cloud_->clear();

  std::vector<rviz::PointCloud::Point> transformed_points;
  rviz::PointCloud::Point default_pt;
  default_pt.color = Ogre::ColourValue(1, 1, 1);
  default_pt.position = Ogre::Vector3::ZERO;

  size_t size = msg.width * msg.height;
  transformed_points.resize(size, default_pt);

  Ogre::Matrix4 transform;

  sensor_msgs::PointCloud2Ptr ptr = boost::make_shared<sensor_msgs::PointCloud2>(msg);
  rviz::XYZPCTransformer xyz;
  if(xyz.supports(ptr)) {
    xyz.transform(ptr, rviz::PointCloudTransformer::Support_XYZ, transform, transformed_points);
  }

  rviz::RGBF32PCTransformer rgb;
  rviz::RGBF32PCTransformer rgb2;
  if(rgb.supports(ptr)) {
    rgb.transform(ptr, rviz::PointCloudTransformer::Support_Color, transform, transformed_points);
  } else if (rgb2.supports(ptr)){
    rgb2.transform(ptr, rviz::PointCloudTransformer::Support_Color, transform, transformed_points);
  } 

  cloud_->addPoints(&(transformed_points.front()), transformed_points.size());
}



}  // namespace viz
}  // namespace objrec