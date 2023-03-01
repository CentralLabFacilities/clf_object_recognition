cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_3d)

message(STATUS ${PCL_LIBRARIES})

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  vision_msgs
  clf_object_recognition_msgs
  clf_object_recognition_cfg
  dynamic_reconfigure
  pcl_conversions
  cv_bridge
)
include_directories(${catkin_INCLUDE_DIRS})
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

catkin_python_setup()

catkin_package()

include_directories(include)
add_executable(
  3d_detector 
  src/node.cpp 
  src/detector.cpp 
  src/clf_object_recognition_3d/img_2_raw_point_cloud_provider.cpp 
  src/clf_object_recognition_3d/load_point_cloud_provider.cpp 
  src/clf_object_recognition_3d/registrated_pcl_provider.cpp 
  src/clf_object_recognition_3d/pcl_to_detection_provider.cpp
  src/Image2Pcl.cpp
  )
target_link_libraries(3d_detector ${catkin_LIBRARIES})

install(TARGETS 3d_detector
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(PROGRAMS
  scripts/simple_3d
  scripts/new_3d
  scripts/detection_via_pcl
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
