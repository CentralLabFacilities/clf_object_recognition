#include <ros/ros.h>
#include "clf_object_recognition_3d/detector.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detect3d");

  ros::NodeHandle nh("detect_3d");

  Detector detect(nh);

  ros::spin();

  ros::shutdown();

  return 0;
}