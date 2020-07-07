#pragma once

#ifndef Q_MOC_RUN
#include <clf_object_recognition_msgs/HullArray.h>

#include "rviz/message_filter_display.h"
#endif

#include <mutex>

namespace objrec
{
namespace viz
{
class HullVisual;

class HullArrayDisplay : public rviz::MessageFilterDisplay<clf_object_recognition_msgs::HullArray>
{
  Q_OBJECT
public:
  HullArrayDisplay();
  ~HullArrayDisplay() override = default;

protected:
  void onInitialize() override;
  void reset() override;

private:
  void processMessage(const clf_object_recognition_msgs::HullArray::ConstPtr& msg) override;

  clf_object_recognition_msgs::HullArray::ConstPtr initMsg_;

  std::mutex mutex_;
  std::vector<std::shared_ptr<HullVisual> > hulls_;
};
}  // namespace viz
}  // namespace objrec
