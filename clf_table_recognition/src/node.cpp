#include <ros/ros.h>
#include "clf_table_recognition/detector.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "table_detect");

  ros::NodeHandle nh("table_detect");

  Detector detect(nh);

  ros::spin();

  ros::shutdown();

  return 0;
}