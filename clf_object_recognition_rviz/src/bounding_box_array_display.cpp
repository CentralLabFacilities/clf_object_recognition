#include "clf_object_recognition_rviz/bounding_box_array_display.h"

#include <QString>
#include <memory>

#include <rviz/properties/bool_property.h>
#include <rviz/properties/float_property.h>

namespace objrec
{
namespace viz
{
BoundingBoxArrayDisplay::BoundingBoxArrayDisplay()
{
}

void BoundingBoxArrayDisplay::onInitialize()
{
  MFDClass::onInitialize();
}

void BoundingBoxArrayDisplay::reset()
{
  MFDClass::reset();
}

void BoundingBoxArrayDisplay::processMessage(const clf_object_recognition_msgs::BoundingBox3DArray::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(mutex_);

  int count = 0;
  // scene_node to map

  for (auto bbox : msg->boxes)
  {
    std::shared_ptr<BoundingBoxVisual> box;
    if (++count > boxes_.size())
    {
      box = std::make_shared<BoundingBoxVisual>(context_->getSceneManager(), scene_node_);
      boxes_.push_back(box);
    }
    else
    {
      box = boxes_.at(count-1);
    }
    box->setMessage(bbox);
  }

  for(int i = boxes_.size() - count; i > 0; i++) {
    boxes_.pop_back();
  }
}

}  // namespace viz
}  // namespace objrec
#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(objrec::viz::BoundingBoxArrayDisplay, rviz::Display)