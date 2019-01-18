cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_merger)

include_directories(include)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  clf_object_recognition_msgs
  vision_msgs
  cv_bridge
)

catkin_package(CATKIN_DEPENDS
  roscpp
  clf_object_recognition_msgs
  vision_msgs
  cv_bridge
)
include_directories(${catkin_INCLUDE_DIRS})



FIND_PACKAGE(OpenCV REQUIRED COMPONENTS 
  highgui 
  core
)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

###########
## Build ##
###########

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++0x")

add_executable(object_merger src/object_merger.cpp) 
target_link_libraries(object_merger ${catkin_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES})
#add_dependencies(object_merger clf_object_recognition_merger)


install(
  TARGETS object_merger
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION} )
