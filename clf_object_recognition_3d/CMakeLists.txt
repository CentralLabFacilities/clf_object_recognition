cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_3d)

find_package(catkin REQUIRED COMPONENTS
  rospy
  vision_msgs
  clf_object_recognition_msgs
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/simple_3d
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
