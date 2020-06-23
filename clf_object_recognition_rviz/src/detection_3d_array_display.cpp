#include "clf_object_recognition_rviz/detection_3d_array_display.h"

#include <QString>
#include <memory>
#include <OGRE/OgreSceneNode.h>

#include <rviz/properties/bool_property.h>
#include <rviz/properties/float_property.h>

namespace objrec
{
namespace viz
{
Detection3DArrayDisplay::Detection3DArrayDisplay()
{
  showLabels_ = new rviz::BoolProperty("show Labels", true, "Draw the Labels.", this, SLOT(slotShowLabels()));
  showProb_ = new rviz::BoolProperty("show Propability", false, "Prefix label with Parent Location", this, SLOT(slotShowProb()));
  showBox_ = new rviz::BoolProperty("show Bounding Box", true, "Prefix label with Parent Location", this, SLOT(slotShowBox()));
  showPoints_ = new rviz::BoolProperty("show Points", false, "Prefix label with Parent Location", this, SLOT(slotShowPoints()));
  labelSize_ = new rviz::FloatProperty("Label size", 0.1, "Character Height of TextLabel", this, SLOT(slotLabelSize()));

}


void Detection3DArrayDisplay::slotLabelSize()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& vis : visuals_)
  {
    vis->setCharacterHeight(labelSize_->getFloat());
  }
}

void Detection3DArrayDisplay::slotShowBox()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& vis : visuals_)
  {
    vis->setShowBox(showBox_->getBool());
  }
}

void Detection3DArrayDisplay::slotShowPoints()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& vis : visuals_)
  {
    vis->setShowPoints(showPoints_->getBool());
  }
}

void Detection3DArrayDisplay::slotShowLabels()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& vis : visuals_)
  {
    vis->setShowLabel(showLabels_->getBool());
  }
}

void Detection3DArrayDisplay::slotShowProb()
{
  std::lock_guard<std::mutex> lock(mutex_);
  for (const auto& vis : visuals_)
  {
    vis->setShowPropability(showProb_->getBool());
  }
}

void Detection3DArrayDisplay::onInitialize()
{
  MFDClass::onInitialize();
}

void Detection3DArrayDisplay::reset()
{
  MFDClass::reset();
}

void Detection3DArrayDisplay::processMessage(const vision_msgs::Detection3DArray::ConstPtr& msg)
{

  ROS_DEBUG("GOT MSG");
  Ogre::Quaternion orientation;
  Ogre::Vector3 position;
  if (!context_->getFrameManager()->getTransform(msg->header.frame_id, msg->header.stamp, position, orientation))
  {
    ROS_DEBUG("Error transforming from frame '%s' to frame '%s'", msg->header.frame_id.c_str(),
              qPrintable(fixed_frame_));
    return;
  }

  scene_node_->setPosition(position);
  scene_node_->setOrientation(orientation);

  {
    std::lock_guard<std::mutex> lock(mutex_);

    int count = 0;
    // scene_node to map

    for (auto detection : msg->detections)
    {
      std::shared_ptr<Detection3DVisual> vis;
      if (++count > visuals_.size())
      {
        vis = std::make_shared<Detection3DVisual>(context_->getSceneManager(), scene_node_);
        
        vis->setShowPropability(showProb_->getBool());
        vis->setShowLabel(showLabels_->getBool());
        vis->setShowPoints(showPoints_->getBool());
        vis->setShowBox(showBox_->getBool());
        vis->setCharacterHeight(labelSize_->getFloat());

        visuals_.push_back(vis);
      }
      else
      {
        vis = visuals_.at(count - 1);
      }
      vis->setMessage(detection);

    }

    for (int i = visuals_.size() - count; i > 0; i--)
    {
      visuals_.pop_back();
    }

  }  // lock_guard mutex_
}

}  // namespace viz
}  // namespace objrec
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(objrec::viz::Detection3DArrayDisplay, rviz::Display)
