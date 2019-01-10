cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_merger)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  clf_object_recognition_msgs
  vision_msgs
  cv_bridge
  tf
)

catkin_package(CATKIN_DEPENDS
  roscpp
  clf_object_recognition_msgs
  vision_msgs
  cv_bridge
  tf
)

FIND_PACKAGE(OpenCV REQUIRED COMPONENTS highgui
                                        core
)

###########
## Build ##
###########

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x")

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(object_merger src/object_merger.cpp) 
target_link_libraries(object_merger ${catkin_LIBRARIES} ${OpenCV_LIBS})
#add_dependencies(object_merger clf_object_recognition_merger)

