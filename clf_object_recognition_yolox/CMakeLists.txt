cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_yolox)

find_package(catkin REQUIRED COMPONENTS
  rospy
  clf_object_recognition_cfg
)

catkin_python_setup()

catkin_package()

install(PROGRAMS
  scripts/yolox_recognition_node
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
