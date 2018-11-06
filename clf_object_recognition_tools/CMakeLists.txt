cmake_minimum_required(VERSION 2.8.12)
project(clf_object_recognition_tools)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rqt_gui
  rqt_gui_py
)

catkin_python_setup()

catkin_package()

install(FILES plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(PROGRAMS
  scripts/annotation_gui
  scripts/test_gui
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
