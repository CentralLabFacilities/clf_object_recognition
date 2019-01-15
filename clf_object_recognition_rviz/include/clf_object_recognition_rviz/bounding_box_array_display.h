#pragma once

#ifndef Q_MOC_RUN
#include <clf_object_recognition_msgs/BoundingBox3DArray.h>

#include "clf_object_recognition_rviz/bounding_box_visual.h"
#include "rviz/message_filter_display.h"
#endif

#include <mutex>

namespace Ogre
{
class SceneNode;
}

namespace rviz
{
class BoolProperty;
class FloatProperty;
}

namespace objrec
{
namespace viz
{
class BoundingBoxArrayDisplay : public rviz::MessageFilterDisplay<clf_object_recognition_msgs::BoundingBox3DArray>
{
  Q_OBJECT
public:
  BoundingBoxArrayDisplay();
  ~BoundingBoxArrayDisplay() override = default;

protected:
  void onInitialize() override;
  void reset() override;

private:
  void processMessage(const clf_object_recognition_msgs::BoundingBox3DArray::ConstPtr& msg) override;

  clf_object_recognition_msgs::BoundingBox3DArray::ConstPtr initMsg_;

  std::mutex mutex_;
  std::vector<std::shared_ptr<BoundingBoxVisual> > boxes_;
};
}  // namespace viz
}  // namespace objrec