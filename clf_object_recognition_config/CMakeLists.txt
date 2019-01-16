cmake_minimum_required(VERSION 2.8.12)

project(clf_object_recognition_config)
find_package(catkin REQUIRED)
SET(CMAKE_BUILD_TYPE distribution)

SET(PREFIX "/usr/local")
IF (CMAKE_INSTALL_PREFIX)
    SET(PREFIX ${CMAKE_INSTALL_PREFIX})
    MESSAGE("-- Found install PREFIX: " ${PREFIX})
ENDIF()

INSTALL(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/config" DESTINATION ${PREFIX}/share/${CMAKE_PROJECT_NAME})
