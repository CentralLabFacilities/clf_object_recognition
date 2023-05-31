#pragma once

#include <string>
#include <ros/ros.h>
#include <ecwm_msgs/ModelVisArray.h>

class ModelProvider
{
public:
  ModelProvider(ros::NodeHandle nh);
  std::string GetModelPath(std::string model_name);
  std::string IDtoObject(int id);
  std::string IDtoModel(int id);

  ecwm_msgs::ModelVisArray latest_models;
  ros::Subscriber model_sub;
  void ModelCallback(const ecwm_msgs::ModelVisArrayPtr& msg);
};