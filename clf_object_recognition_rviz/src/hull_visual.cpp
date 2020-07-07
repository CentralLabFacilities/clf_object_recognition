#include <OGRE/OgreManualObject.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>

#include <rviz/properties/parse_color.h>

#include "clf_object_recognition_rviz/hull_visual.h"

namespace objrec
{
namespace viz
{
HullVisual::HullVisual(Ogre::SceneManager* scene_manager, Ogre::SceneNode* parent_node)
{
  scene_manager_ = scene_manager;

  object_node_ = parent_node->createChildSceneNode();
  manual_object_ = scene_manager->createManualObject();
  manual_object_->setDynamic(true);
  object_node_->attachObject(manual_object_);

  hull_.reset(new rviz::BillboardLine( scene_manager, object_node_ ));
}

HullVisual::~HullVisual()
{
  scene_manager_->destroySceneNode(object_node_);
}

void HullVisual::addPose(Eigen::Vector3d& p)
{
  manual_object_->position(p[0], p[1], p[2]);
}

// ----------------------------------------------------------------------------
void HullVisual::setMessage(const clf_object_recognition_msgs::Hull msg)
{
  hull_->clear();
  hull_->setMaxPointsPerLine(2);
  hull_->setNumLines(msg.polygon.points.size() * 3);
  hull_->setLineWidth(0.01);

  for(unsigned int i = 0; i < msg.polygon.points.size(); ++i)
    {        
      int j = (i + 1) % msg.polygon.points.size();

      float x1 = msg.polygon.points[i].x;
      float x2 = msg.polygon.points[j].x;

      float y1 = msg.polygon.points[i].y;
      float y2 = msg.polygon.points[j].y;

      // Low line
      if (i != 0)
          hull_->newLine();
      hull_->addPoint(Ogre::Vector3(x1, y1, msg.z_min));
      hull_->addPoint(Ogre::Vector3(x2, y2, msg.z_min));

      // High line
      hull_->newLine();
      hull_->addPoint(Ogre::Vector3(x1, y1, msg.z_max));
      hull_->addPoint(Ogre::Vector3(x2, y2, msg.z_max));

      // Vertical line
      hull_->newLine();
      hull_->addPoint(Ogre::Vector3(x1, y1, msg.z_min));
      hull_->addPoint(Ogre::Vector3(x1, y1, msg.z_max));
    }

  Ogre::ColourValue color = rviz::qtToOgre(color_);
  color.a = 1;
  hull_->setColor(color.r, color.g, color.b, color.a);

}

}  // namespace viz
}  // namespace objrec
