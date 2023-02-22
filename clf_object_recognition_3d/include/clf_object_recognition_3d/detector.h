#pragma once

#include <ros/node_handle.h>     // for NodeHandle
#include <ros/service_server.h>  // for ServiceServer
#include <ros/service_client.h>

#include <dynamic_reconfigure/server.h>
#include <clf_object_recognition_cfg/Detect3dConfig.h>

#include <clf_object_recognition_msgs/Detect3D.h>

class Detector  {

public:
    Detector(ros::NodeHandle nh);
    
private:
    ros::NodeHandle nh_;

    clf_object_recognition_cfg::Detect3dConfig config;
    dynamic_reconfigure::Server<clf_object_recognition_cfg::Detect3dConfig> reconfigure_server;
    void ReconfigureCallback(const clf_object_recognition_cfg::Detect3dConfig& config, uint32_t level);

    bool ServiceDetect3D(clf_object_recognition_msgs::Detect3D::Request& req,  clf_object_recognition_msgs::Detect3D::Response& res);
    ros::ServiceServer srv_detect_3d;

    ros::ServiceClient srv_detect2d;

};