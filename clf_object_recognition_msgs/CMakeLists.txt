cmake_minimum_required(VERSION 2.8.12) 
project(clf_object_recognition_msgs)

set(MSG_DEPS) #geometry_msgs actionlib_msgs

find_package(catkin REQUIRED COMPONENTS message_generation ${ MSG_DEPS })

#ROS messages, services and actions
#add_message_files(DIRECTORY msg FILES
#)

#add_service_files(DIRECTORY srv FILES
#)

#add_action_files(DIRECTORY action FILES
#)

generate_messages(DEPENDENCIES ${ MSG_DEPS })

catkin_package(CATKIN_DEPENDS message_runtime ${ MSG_DEPS })
