#pragma once

#include <clf_object_recognition_msgs/Hull.h>

#include <QColor>
#include <tf2_eigen/tf2_eigen.h>

#include <rviz/ogre_helpers/billboard_line.h>

namespace Ogre
{
class Entity;
class Vector3;
class Quaternion;
class ManualObject;
class SceneManager;
class SceneNode;
}  // namespace Ogre

namespace rviz
{
class MovableText;
}

namespace objrec
{
namespace viz
{
class HullVisual
{
public:
  HullVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node);
  ~HullVisual();

  // Configure the visual to show the data in the message.
  void setMessage(const clf_object_recognition_msgs::Hull msg);

private:
  void addPose(Eigen::Vector3d& p);

  Ogre::SceneNode* object_node_{ nullptr };
  Ogre::SceneManager* scene_manager_{ nullptr };

  Ogre::ManualObject* manual_object_{ nullptr };
  std::shared_ptr<rviz::BillboardLine> hull_;

  QColor color_{ QColor(0, 255, 0) };
};
}  // namespace viz
}  // namespace objrec