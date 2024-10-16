#pragma once

#ifndef Q_MOC_RUN
#include <vision_msgs/Detection3DArray.h>

#include "rviz/message_filter_display.h"
#endif

#include <mutex>

namespace rviz
{
class BoolProperty;
class FloatProperty;
}  // namespace rviz

namespace objrec
{
namespace viz
{
class Detection3DVisual;

class Detection3DArrayDisplay : public rviz::MessageFilterDisplay<vision_msgs::Detection3DArray>
{
  Q_OBJECT
public:
  Detection3DArrayDisplay();
  virtual ~Detection3DArrayDisplay();

protected:
  virtual void onInitialize() override;
  virtual void reset() override;

private Q_SLOTS:
  void slotShowLabels();
  void slotShowBox();
  void slotShowPoints();
  void slotShowProb();
  void slotLabelSize();

private:
  virtual void processMessage(const vision_msgs::Detection3DArray::ConstPtr& msg) override;

  vision_msgs::Detection3DArray::ConstPtr initMsg_;

  std::mutex mutex_;
  std::vector<std::shared_ptr<Detection3DVisual> > visuals_;

  rviz::BoolProperty* showLabels_;
  rviz::BoolProperty* showProb_;
  rviz::BoolProperty* showBox_;
  rviz::BoolProperty* showPoints_;
  rviz::FloatProperty* labelSize_;
};
}  // namespace viz
}  // namespace objrec
