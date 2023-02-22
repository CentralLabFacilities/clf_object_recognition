#include "clf_object_recognition_3d/detector.h"

#include <clf_object_recognition_msgs/Detect2D.h>

Detector::Detector(ros::NodeHandle nh) {
    srv_detect2d = nh.serviceClient<clf_object_recognition_msgs::Detect2D>("/yolox/recognize");
    srv_detect_3d = nh.advertiseService("/simple_detections", &Detector::ServiceDetect3D, this);

    auto f = [this](auto&& PH1, auto&& PH2) { ReconfigureCallback(PH1, PH2); };
    reconfigure_server.setCallback(f);
    ros::spinOnce();

}

void Detector::ReconfigureCallback(const clf_object_recognition_cfg::Detect3dConfig& input, uint32_t /*level*/)
{
    ROS_INFO_NAMED("detector", "Reconfigure");
    config = input;
}
    
bool Detector::ServiceDetect3D(clf_object_recognition_msgs::Detect3D::Request& req, clf_object_recognition_msgs::Detect3D::Response& res) {

    ROS_INFO_STREAM_NAMED("detector", "ServiceDetect3D() called " << req);

    /*clf_object_recognition_msgs::Detect2D param;
    auto ok = srv_detect2d.call(param);
    if(!ok) {
        ROS_ERROR_STREAM_NAMED("detector", "cant call detections ");
        return false;
    }*/

    for(auto& detection : param.response.detections) {

    }

    if(config.publish_detections) ROS_INFO_STREAM_NAMED("detector", "publishing detections ");


    ROS_INFO_STREAM_NAMED("detector", "got detections " << param.response);

    return true;

}