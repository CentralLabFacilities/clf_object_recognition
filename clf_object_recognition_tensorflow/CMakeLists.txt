cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_tensorflow)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/object_recognition_node
  scripts/object_detection_node
  scripts/retrain_inception
  scripts/train_detector
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
