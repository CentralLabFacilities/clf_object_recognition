#include "clf_object_recognition_rviz/detection_3d_array_display.h"

int main(int argc, char** argv)  // NOLINT
{
    ros::init(argc, argv, "link_test");

    objrec::viz::Detection3DArrayDisplay disp{};
    return 0;

}