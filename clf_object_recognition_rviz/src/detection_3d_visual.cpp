#include "clf_object_recognition_rviz/detection_3d_visual.h"

#include <OGRE/OgreManualObject.h>
#include <OGRE/OgreMaterialManager.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreSharedPtr.h>
#include <OGRE/OgreTechnique.h>
#include <OGRE/OgreTextureManager.h>
#include <OGRE/OgreEntity.h>
#include <OGRE/OgreQuaternion.h>

#include <tf2_eigen/tf2_eigen.h>
#include <eigen_conversions/eigen_msg.h>

#include <QColor>

#include <ros/ros.h>

#include <rviz/ogre_helpers/movable_text.h>
#include <rviz/properties/parse_color.h>
#include <rviz/mesh_loader.h>
#include <rviz/geometry.h>

#include <memory>



namespace objrec
{
namespace viz
{
Detection3DVisual::Detection3DVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node)
{
  scene_manager_ = scene_manager;

  main_node_ = parent_node->createChildSceneNode();

  bb_node_ = main_node_->createChildSceneNode();
  text_node_ = main_node_->createChildSceneNode();
  points_node_ = main_node_->createChildSceneNode();

  bbox_ = std::make_unique<BoundingBoxVisual>(scene_manager, bb_node_);
  text_ = new rviz::MovableText("unknown object - 0.0");
  text_->setTextAlignment(rviz::MovableText::H_LEFT, rviz::MovableText::V_CENTER);
  text_node_->attachObject(text_);

}

Detection3DVisual::~Detection3DVisual()
{
  scene_manager_->destroySceneNode(bb_node_);
  scene_manager_->destroySceneNode(text_node_);
  scene_manager_->destroySceneNode(points_node_);
  scene_manager_->destroySceneNode(main_node_);
}

void Detection3DVisual::setShowBox(const bool show) {
  bb_node_->setVisible(show);
}

void Detection3DVisual::setShowLabel(const bool show)
{
  text_node_->setVisible(show);
}

void Detection3DVisual::setCharacterHeight(const float size)
{
  text_size_ = size;
  if (text_ != nullptr)
  {
    text_->setCharacterHeight(text_size_);
  }
}

void Detection3DVisual::setShowPoints(const bool show) {
  points_node_->setVisible(show);
}

void Detection3DVisual::setShowPropability(const bool show)
{
  show_prob_ = show;
  updateLabel();
}


// ----------------------------------------------------------------------------
void Detection3DVisual::setMessage(const vision_msgs::Detection3D msg)
{
  bbox_->setMessage(msg.bbox);
  updateHypothesis(msg);
  updateLabel();
  auto center = msg.bbox.center.position;
  auto size = msg.bbox.size;
  Ogre::Vector3 position(center.x, center.y, center.z - size.z / 2);
  text_node_->setPosition(position);
}

void Detection3DVisual::updateLabel() {
  std::ostringstream out;
  out.precision(2);
  out << std::fixed << cur_prob_;
  auto txt = cur_hyp_ + ((show_prob_) ? " - " + out.str() : "");
  text_->setCaption(txt);
  ROS_INFO_STREAM("caption:" << txt);
}

template <template<class,class,class...> class C, typename K, typename V, typename... Args>
V GetWithDef(const C<K,V,Args...>& m, K const& key, const V & defval)
{
    typename C<K,V,Args...>::const_iterator it = m.find( key );
    if (it == m.end())
        return defval;
    return it->second;
}

std::string Detection3DVisual::getName(int id, std::string fixed_name)
{
  if(fixed_name != "") {
    return fixed_name;
  }
  std::map<std::string,std::string> labels;
  ros::param::get(std::string("/object_labels"), labels);
  return GetWithDef(labels,std::to_string(id),std::string("unknown"));
}

void Detection3DVisual::updateHypothesis(const vision_msgs::Detection3D msg)
{
  if(!msg.results.empty()) {
    cur_prob_ = msg.results.front().score;
    cur_hyp_ = getName(msg.results.front().id);
  } else {
    cur_prob_ = 0.0;
    cur_hyp_ = "Unknown";
  }
}

}  // namespace viz
}  // namespace objrec
