cmake_minimum_required(VERSION 3.16)
project(clf_table_recognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_compile_options(-fsanitize=undefined)
#add_link_options(-fsanitize=undefined)

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0.0)
        message(FATAL_ERROR "GCC version must be at least 9 you version ${CMAKE_CXX_COMPILER_VERSION} has filesystem under experimental and requires manual linking")
endif()


find_package(catkin REQUIRED COMPONENTS
  roscpp
  vision_msgs
  clf_object_recognition_msgs
  dynamic_reconfigure
  pcl_conversions
  pcl_ros
  cv_bridge
  geometry_msgs
  geometric_shapes
  ecwm_msgs
  eigen_conversions
  image_geometry
  tf2_ros
  tf2_eigen

  tf2_geometry_msgs
  tf2_sensor_msgs

  pcl_conversions
  pcl_ros

  image_transport
  
)
include_directories(${catkin_INCLUDE_DIRS})
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

catkin_package()

include_directories(BEFORE include)

add_executable(table_detector 
  src/node.cpp 
  src/detector.cpp
)

target_link_libraries(table_detector ${catkin_LIBRARIES})

install(TARGETS table_detector
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
