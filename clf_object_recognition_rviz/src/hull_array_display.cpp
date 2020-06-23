#include "clf_object_recognition_rviz/hull_array_display.h"

#include <QString>
#include <memory>
#include <OGRE/OgreSceneNode.h>

#include <rviz/properties/bool_property.h>
#include <rviz/properties/float_property.h>

namespace objrec
{
namespace viz
{
HullArrayDisplay::HullArrayDisplay()
{
}

void HullArrayDisplay::onInitialize()
{
  MFDClass::onInitialize();
}

void HullArrayDisplay::reset()
{
  MFDClass::reset();
}

void HullArrayDisplay::processMessage(const clf_object_recognition_msgs::HullArray::ConstPtr& msg)
{
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

    for (auto hullmsg : msg->hulls)
    {
      std::shared_ptr<HullVisual> hull;
      if (++count > hulls_.size())
      {
        hull = std::make_shared<HullVisual>(context_->getSceneManager(), scene_node_);
        hulls_.push_back(hull);
      }
      else
      {
        hull = hulls_.at(count - 1);
      }
      hull->setMessage(hullmsg);
    }

    for (int i = hulls_.size() - count; i > 0; i--)
    {
      hulls_.pop_back();
    }
  }  // lock_guard mutex_
}

}  // namespace viz
}  // namespace objrec
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(objrec::viz::HullArrayDisplay, rviz::Display)
