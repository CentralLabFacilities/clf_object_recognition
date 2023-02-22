cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_3d)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  vision_msgs
  clf_object_recognition_msgs
  clf_object_recognition_cfg
  dynamic_reconfigure
)
include_directories(${catkin_INCLUDE_DIRS})

catkin_python_setup()

catkin_package()

include_directories(include)
add_executable(3d_detector src/node.cpp src/detector.cpp)
target_link_libraries(3d_detector ${catkin_LIBRARIES})

install(TARGETS 3d_detector
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(PROGRAMS
  scripts/simple_3d
  scripts/new_3d
  scripts/pcl_registration
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
