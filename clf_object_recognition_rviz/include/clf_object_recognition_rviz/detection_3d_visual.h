#pragma once

#include "vision_msgs/Detection3D.h"
#include <memory>  // for unique_ptr
#include <string>  // for string

#include "clf_object_recognition_rviz/bounding_box_visual.h"

#include "clf_object_recognition_rviz/point_cloud_visual.h"

namespace Ogre
{
class SceneManager;
class SceneNode;
}  // namespace Ogre

namespace rviz
{
class MovableText;
class PointCloudCommon;
class DisplayContext;
}  // namespace rviz

namespace objrec
{
namespace viz
{
class Detection3DVisual
{
public:
  Detection3DVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node);
  ~Detection3DVisual();

  void initializePointCommon(rviz::DisplayContext* context);

  // Configure the visual to show the data in the message.
  void setMessage(const vision_msgs::Detection3D msg);

  void setShowBox(bool b);
  void setShowLabel(const bool show);
  void setCharacterHeight(const float size);
  void setShowPoints(const bool show);
  void setShowPropability(const bool show);

protected:
  std::string getName(int id, std::string fixed_name = "");
  std::string getName(const std::string& id, std::string fixed_name = "");
  void updateHypothesis(const vision_msgs::Detection3D msg);
  void updateCloud(const sensor_msgs::PointCloud2& msg);
  void updateLabel();

private:
  bool show_prob_{ true };
  std::string cur_hyp_{ "unknown" };
  double cur_prob_{ 0.0 };
  float text_size_;

  rviz::PointCloudCommon* point_cloud_common_;

  Ogre::SceneManager* scene_manager_{ nullptr };
  Ogre::SceneNode* main_node_{ nullptr };
  Ogre::SceneNode* bb_node_{ nullptr };
  Ogre::SceneNode* text_node_{ nullptr };
  Ogre::SceneNode* points_node_{ nullptr };

  rviz::MovableText* text_{ nullptr };

  std::unique_ptr<BoundingBoxVisual> bbox_{ nullptr };

  std::unique_ptr<PointCloudVisual> pc_{ nullptr };
};
}  // namespace viz
}  // namespace objrec
