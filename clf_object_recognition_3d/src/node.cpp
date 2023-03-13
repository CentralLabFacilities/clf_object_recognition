#include <ros/ros.h>
#include "detector.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "detect3d");

  ros::NodeHandle nh("detect3d");

  Detector detect(nh);

  ros::spin();
 
  ros::shutdown();

  return 0;
}