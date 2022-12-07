#pragma once

#ifndef Q_MOC_RUN
#include <vision_msgs/BoundingBox3DArray.h>

#include "rviz/message_filter_display.h"
#endif

#include <mutex>

namespace objrec
{
namespace viz
{
class BoundingBoxVisual;

class BoundingBoxArrayDisplay : public rviz::MessageFilterDisplay<vision_msgs::BoundingBox3DArray>
{
  Q_OBJECT
public:
  BoundingBoxArrayDisplay();
  ~BoundingBoxArrayDisplay() override = default;

protected:
  void onInitialize() override;
  void reset() override;

private:
  void processMessage(const vision_msgs::BoundingBox3DArray::ConstPtr& msg) override;

  vision_msgs::BoundingBox3DArray::ConstPtr initMsg_;

  std::mutex mutex_;
  std::vector<std::shared_ptr<BoundingBoxVisual> > boxes_;
};
}  // namespace viz
}  // namespace objrec
