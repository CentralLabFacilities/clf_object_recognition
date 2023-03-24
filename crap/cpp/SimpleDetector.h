#include <ros/ros.h>
// ros messages
#include "vision_msgs/Detection3D.h"
#include "vision_msgs/Detection3DArray.h"
// ros services
#include 
#include <string>

namespace clf_object_recognition_3d {
    class SimpleDetector {
        public:
            SimpleDetector(const std::string& serviceName, bool publishDetections = true);
            srv::Detect3DResponse callbackDetect3D();
        private:
            //getDetectors();
    }
}