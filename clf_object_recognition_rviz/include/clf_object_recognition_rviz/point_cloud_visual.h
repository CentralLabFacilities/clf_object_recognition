#pragma once

#include "sensor_msgs/PointCloud2.h"

#include "rviz/ogre_helpers/point_cloud.h"

namespace Ogre
{
class SceneManager;
class SceneNode;
}  // namespace Ogre

namespace objrec
{
namespace viz
{
class PointCloudVisual
{
public:
  PointCloudVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node);
  ~PointCloudVisual();

  // Configure the visual to show the data in the message.
  void setMessage(const sensor_msgs::PointCloud2& msg);

  Ogre::SceneNode* object_node_{ nullptr };
  Ogre::SceneManager* scene_manager_{ nullptr };
  boost::shared_ptr<rviz::PointCloud> cloud_;

private:
};
}  // namespace viz
}  // namespace objrec