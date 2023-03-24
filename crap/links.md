- poitclounds: https://pointclouds.org/
- http://wiki.ros.org/message_filters - Time Synchronizer


## To Do

# Goal

write a rosservice
produce vision_msgs.msg import Detection3D, Detection3DArray
estimate: given an image -> produce a 6D Pose Estimation

rosservice is called, waits for images, delivers a 6D Pose Estimation for each object.


# Further goals

instead of an image as argument subscribe to rgb and also to depth image, synchronise topics and use callback result image for the estimate function.

Instead of raw pixels use 
detections = self._get_detections(img)
for d2d in detections.detections:
	....
estimate: 2d-detections -> 6d-pose-estimation


# Relevant code

processing file
/home/lvonseelstrang/RoboCup/clf_object_recognition/clf_object_recognition_3d/src/clf_object_recognition_3d

launch file:
/home/lvonseelstrang/RoboCup/clf_object_recognition/clf_object_recognition_3d/launch


display mesh:
http://wiki.ros.org/rviz/DisplayTypes/Marker#Mesh_Resource_.28MESH_RESOURCE.3D10.29_.5B1.1.2B-.5D


# useful commands

getting the object corresponding to an id
rosparam get /object_labels/id
