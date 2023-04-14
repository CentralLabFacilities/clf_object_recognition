cmake_minimum_required(VERSION 3.16)
project(clf_object_recognition_3d)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fsanitize=undefined)
add_link_options(-fsanitize=undefined)

if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0.0)
        message(FATAL_ERROR "GCC version must be at least 9 you version ${CMAKE_CXX_COMPILER_VERSION} has filesystem under experimental and requires manual linking")
endif()


find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  vision_msgs
  clf_object_recognition_msgs
  clf_object_recognition_cfg
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
  
)
include_directories(${catkin_INCLUDE_DIRS})
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

catkin_python_setup()

catkin_package()

include_directories(BEFORE include)

add_executable(3d_detector 
  src/node.cpp 
  src/detector.cpp 
  src/model_provider.cpp
  src/cloud_from_image.cpp
)

target_link_libraries(3d_detector ${catkin_LIBRARIES})

install(TARGETS 3d_detector
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


install(PROGRAMS
  scripts/simple_3d
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
