#include "clf_object_recognition_rviz/point_cloud_visual.h"

#include "sensor_msgs/PointCloud2.h"
#include "rviz/default_plugin/point_cloud_transformers.h"

#include <rviz/default_plugin/point_cloud_common.h>
#include <rviz/default_plugin/point_cloud_transformers.h>
#include <rviz/display_context.h>
#include <rviz/frame_manager.h>
#include <rviz/ogre_helpers/point_cloud.h>
#include <rviz/properties/int_property.h>
#include <rviz/validate_floats.h>

#include <boost/make_shared.hpp>

namespace objrec
{
namespace viz
{
inline int32_t findChannelIndex(const sensor_msgs::PointCloud2& cloud, const std::string& channel)
{
  for (size_t i = 0; i < cloud.fields.size(); ++i)
  {
    if (cloud.fields[i].name == channel)
    {
      return i;
    }
  }

  return -1;
}



PointCloudVisual::PointCloudVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node)
{
  scene_manager_ = scene_manager;
  object_node_ = parent_node->createChildSceneNode();

  cloud_.reset(new rviz::PointCloud());
  cloud_->setRenderMode(rviz::PointCloud::RenderMode::RM_POINTS);

  object_node_->attachObject(cloud_.get());
}
PointCloudVisual::~PointCloudVisual()
{
}

void do_nothing_deleter(int *)
{
    return;
}


void PointCloudVisual::setMessage(const sensor_msgs::PointCloud2& cloud)
{

  // Filter any nan values out of the cloud.  Any nan values that make it through to PointCloudBase
  // will get their points put off in lala land, but it means they still do get processed/rendered
  // which can be a big performance hit
  sensor_msgs::PointCloud2Ptr filtered(new sensor_msgs::PointCloud2);
  int32_t xi = findChannelIndex(cloud, "x");
  int32_t yi = findChannelIndex(cloud, "y");
  int32_t zi = findChannelIndex(cloud, "z");

  if (xi == -1 || yi == -1 || zi == -1)
  {
    return;
  }

  const uint32_t xoff = cloud.fields[xi].offset;
  const uint32_t yoff = cloud.fields[yi].offset;
  const uint32_t zoff = cloud.fields[zi].offset;
  const uint32_t point_step = cloud.point_step;
  const size_t point_count = cloud.width * cloud.height;

  if (point_count * point_step != cloud.data.size())
  {
    return;
  }

  filtered.data.resize(cloud.data.size());
  uint32_t output_count;
  if (point_count == 0)
  {
    output_count = 0;
  }
  else
  {
    uint8_t* output_ptr = &filtered->data.front();
    const uint8_t *ptr = &cloud.data.front(), *ptr_end = &cloud.data.back(), *ptr_init;
    size_t points_to_copy = 0;
    for (; ptr < ptr_end; ptr += point_step)
    {
      float x = *reinterpret_cast<const float*>(ptr + xoff);
      float y = *reinterpret_cast<const float*>(ptr + yoff);
      float z = *reinterpret_cast<const float*>(ptr + zoff);
      if (rviz::validateFloats(x) && rviz::validateFloats(y) && rviz::validateFloats(z))
      {
        if (points_to_copy == 0)
        {
          // Only memorize where to start copying from
          ptr_init = ptr;
          points_to_copy = 1;
        }
        else
        {
          ++points_to_copy;
        }
      }
      else
      {
        if (points_to_copy)
        {
          // Copy all the points that need to be copied
          memcpy(output_ptr, ptr_init, point_step * points_to_copy);
          output_ptr += point_step * points_to_copy;
          points_to_copy = 0;
        }
      }
    }
    // Don't forget to flush what needs to be copied
    if (points_to_copy)
    {
      memcpy(output_ptr, ptr_init, point_step * points_to_copy);
      output_ptr += point_step * points_to_copy;
    }
    output_count = (output_ptr - &filtered->data.front()) / point_step;
  }

  filtered->header = cloud->header;
  filtered->fields = cloud->fields;
  filtered->data.resize(output_count * point_step);
  filtered->height = 1;
  filtered->width = output_count;
  filtered->is_bigendian = cloud->is_bigendian;
  filtered->point_step = point_step;
  filtered->row_step = output_count;

  // */
  //auto filtered = msg;

  cloud_->clear();

  std::vector<rviz::PointCloud::Point> transformed_points;
  rviz::PointCloud::Point default_pt;
  default_pt.color = Ogre::ColourValue(1, 1, 1);
  default_pt.position = Ogre::Vector3::ZERO;

  size_t size = msg.width * msg.height;
  transformed_points.resize(size, default_pt);

  Ogre::Matrix4 transform;

  sensor_msgs::PointCloud2Ptr ptr = boost::make_shared<sensor_msgs::PointCloud2>(filtered);
  rviz::XYZPCTransformer xyz;
  if (xyz.supports(ptr))
  {
    xyz.transform(ptr, rviz::PointCloudTransformer::Support_XYZ, transform, transformed_points);
  }

  rviz::RGBF32PCTransformer rgb;
  rviz::RGBF32PCTransformer rgb2;
  if (rgb.supports(ptr))
  {
    rgb.transform(ptr, rviz::PointCloudTransformer::Support_Color, transform, transformed_points);
  }
  else if (rgb2.supports(ptr))
  {
    rgb2.transform(ptr, rviz::PointCloudTransformer::Support_Color, transform, transformed_points);
  }



  cloud_->addPoints(&(transformed_points.front()), transformed_points.size());
}

}  // namespace viz
}  // namespace objrec