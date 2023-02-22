# ros
import rospy

#msgs
from vision_msgs.msg import Detection3D, Detection3DArray #, ObjectHypothesis3D

import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2


#srv
from clf_object_recognition_msgs.srv import Detect3D, Detect2DImage, Detect3DResponse

# openCV and pcl
import cv2
from cv_bridge import CvBridge
import pcl
import pcl_ros
from pcl import VoxelGrid, StatisticalOutlierRemoval

# for load_pcl_for_classification
import os
import pyvista as pv


# for object_pcl_callback
from tf.transformations import quaternion_from_matrix
from geometry_msgs.msg import PoseStamped, Quaternion


# for pcl_to_detection
from sensor_msgs import point_cloud2 as pcl2
import numpy as np
from geometry_msgs.msg import Pose, Vector3


# for pcl_2_gripper_pos
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion
from tf.transformations import quaternion_matrix, euler_from_matrix, quaternion_from_matrix, translation_matrix


class Image2Pcl():

    def __init__(self, detect_2d_topic, publish_detections=True, publish_raw_pcl=True, publish_clean_pcl=True):

        self.publish_detections = publish_detections
        self.publish_raw_pcl = publish_raw_pcl
        self.publish_clean_pcl = publish_clean_pcl
        
        self.srv_detect = rospy.ServiceProxy(detect_2d_topic, Detect2DImage)
        #self.service = rospy.Service("simple_detect", Detect3D, self.callback_detect_3d)

        #self.srv_detect = rospy.ServiceProxy(detect3d_topic, Detect3DImage)
        #self.service = rospy.Service("simple_detect", Detect3D, self.service_callback_pcl_pub)
        self.service = rospy.Service("pcl_registration", Detect3D, self.service_callback_pcl_pub)

        # Initialize publishers
        if self.publish_detections:
            self.detections_pub = rospy.Publisher('/simple_detections', Detection3DArray, queue_size=10)
        if self.publish_raw_pcl:
            self.raw_pcl_pub = rospy.Publisher('/raw_object_pcl', PointCloud2, queue_size=1)
        if self.publish_clean_pcl:
            self.clean_object_pcl_pub = rospy.Publisher('/clean_object_pcl', PointCloud2, queue_size=1)

        # Initialize subscribers
        self.image_sub = message_filters.Subscriber('/xtion/rgb/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/xtion/depth_registered/image_raw', Image)
        self.info_sub = message_filters.Subscriber('/xtion/depth_registered/camera_info', CameraInfo)

        # Synchronize image and camera info topics
        self.ts = message_filters.TimeSynchronizer([self.image_sub, self.depth_sub, self.info_sub], 10)
        self.ts.registerCallback(self.callback)

        #self.ats = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.depth_sub, self.info_sub], 10, 0.1)
        #self.ats.registerCallback(self.callback_pcl_pub)

        # self.object_pcl_sub = rospy.Subscriber('/raw_object_pcl', PointCloud2, self.object_pcl_callback)
        # self.gripper_target = rospy.Subscriber('/clean_object_pcl', PointCloud2, self.pcl_2_gripper_pos)


    def callback(self, image, depth, camera_info):
        self.image = image
        self.depth = depth
        self.camera_info = camera_info

    
    def _get_detections(self, img):
        try:
            result = self.srv_detect(img)
            return result
        except Exception as e:
            raise rospy.ServiceException(e)


    # Define callback function for synchronized image streams
    def service_callback_pcl_pub(self, req): #, rgb_msg, depth_msg, info_msg):
        # Convert RGB image to cv2 image
        #bridge = CvBridge()
        #rgb_img = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')

        resp = Detect3DResponse()

        # get image
        img = self.image
        depth_img = self.depth
        info_msg = self.camera_info
        detections = self._get_detections(img)

        for d2d in detections.detections:
            class_name = d2d.results[0].class_name
            certainty = d2d.results[0].score

            xc = d2d.bbox.center.x
            yc = d2d.bbox.center.y
            width = d2d.bbox.size_x
            height = d2d.bbox.size_y

            # Extract bounding box coordinates
            # using the output of the object classifier to get the bounding box coordinates
            xmin = int(xc - width/2)
            ymin = int(yc - height/2)
            xmax = int(xc + width/2)
            ymax = int(yc + height/2)

            # print("detection: ", d2d)

            # Crop RGB and depth images
            #rgb_crop = rgb_img[ymin:ymax, xmin:xmax]
            #depth_crop = depth_msg[ymin:ymax, xmin:xmax]
            rgb_crop = self.image[ymin:ymax, xmin:xmax]
            depth_crop = self.depth[ymin:ymax, xmin:xmax]

            # Convert depth image to point cloud
            pcl = pcl_ros.point_cloud2.create_cloud_xyz32(info_msg.header, depth_crop)

            # Apply voxel grid filter
            voxel_grid = VoxelGrid()
            voxel_grid.set_leaf_size(0.01, 0.01, 0.01)
            pcl_filtered = voxel_grid.filter(pcl)

            # Apply statistical outlier removal filter
            sor = StatisticalOutlierRemoval()
            sor.set_mean_k(50)
            sor.set_std_dev_mul_thresh(1.0)
            pcl_filtered = sor.filter(pcl_filtered)

            if self.publish_raw_pcl:
            # Publish filtered point cloud
                self.raw_pcl_pub.publish(pcl_filtered)

            if self.publish_clean_pcl:
                clean_object_pcl_msg = self.object_pcl_2_clean_pcl(self, pcl_filtered, class_name)
                self.clean_object_pcl_pub.publish(clean_object_pcl_msg)

            if self.publish_detections:
                score = certainty
                if clean_object_pcl_msg is not None:
                    detection = self.pcl_to_detection(clean_object_pcl_msg, class_name, score)
                else:
                    detection = self.pcl_to_detection(pcl_filtered, class_name, score)

                detection.header = d2d.header
                detection.results = d2d.results
                
                resp.detections.append(detection)
                #self.detections_pub.publish(detection)
                

        if len(detections.detections) > 0 and self.publish_detections:
            msg = Detection3DArray()
            msg.header = detections.detections.header
            msg.detections = resp.detections
            self.detections_pub.publish(msg)

        return resp
    

    def load_pcl_for_classification(classification, models_path, num_points=1000):
        """
        classification (String) -- name of the classified class. 
        e.g. 009_gelatin_box will look for the folder gelatin_box.

        All meshes should be converted into pcl importable files once and should be safed. 

        For now this conversion happens on demand.
        """
        models_dir = models_path
        subdirs = [name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name))]

        matchin_dir = None
        for subdir in subdirs:
            for classification in subdir:
                matching_dir = os.path.join(models_dir, subdir)
                break
        
        dae_file = os.path.join(matchin_dir, 'meshes', 'textured.dae')
        # Load the DAE file
        mesh = pv.read(dae_file)
        # Convert the mesh to a point cloud - compute_normals=False computation of normals can be expensive
        point_cloud = mesh.sample(n=num_points, random=True, compute_normals=False)
        return point_cloud



    # Define callback function for object point cloud
    def object_pcl_2_clean_pcl(self, pcl_msg, class_name, models_path=None):
        # Load 3D object data
        if models_path is None:
            models_path = "../../../../gazebo_ycb/models/"
            models_path = "/home/lvonseelstrang/RoboCup/gazebo_ycb/models/"
        
        try:    
            object_cloud = self.load_pcl_for_classification(classification=class_name, models_path=models_path)
        except:
            object_cloud = pcl_msg
            str = "Could not load the reference 3D Object. The reference should be located in: " + models_path + "/<class_name>/meshes/textured.dae\nContinueing with the perceived pcl."
            rospy.loginfo(str)
            return object_cloud

        # Apply voxel grid filter to 3D object point cloud
        voxel_grid = pcl.VoxelGrid()
        voxel_grid.set_leaf_size(0.01, 0.01, 0.01)
        object_cloud_filtered = voxel_grid.filter(object_cloud)

        # Convert object point cloud to ROS message
        object_pcl_msg = pcl_ros.point_cloud2.create_cloud_xyz32(pcl_msg.header, object_cloud_filtered.to_array())

        # Register object point cloud with estimated object point cloud
        icp = pcl.IterativeClosestPoint()
        icp.setInputSource(object_cloud_filtered)
        icp.setInputTarget(pcl_msg)
        converged, transf, estimate, fitness = icp.align(object_cloud_filtered)

        # Publish registered point cloud
        if converged:
            # Convert transformation matrix to ROS message
            pose = PoseStamped()
            pose.header.stamp = rospy.Time.now()
            pose.header.frame_id = pcl_msg.header.frame_id
            pose.pose.position.x = transf[0, 3]
            pose.pose.position.y = transf[1, 3]
            pose.pose.position.z = transf[2, 3]
            q = quaternion_from_matrix(transf)
            pose.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
            clean_object_pcl_msg = pcl_ros.transform_cloud(object_pcl_msg, pose)

            return clean_object_pcl_msg
            # self.clean_object_pcl_pub.publish(clean_object_pcl_msg)



    def pcl_to_detection(pcl_msg, class_name, score):
        # convert pcl_msg to numpy array
        point_cloud = pcl2.read_points(pcl_msg)
        points = np.array(list(point_cloud), dtype=np.float32)

        # create new Detection3D message
        detection = Detection3D()

        # set the results field based on the object classification
        results = ObjectHypothesis3D()
        results.label = class_name
        results.score = score
        detection.results.append(results)

        # set the bounding box field based on the point cloud
        min_point, max_point = np.min(points, axis=0), np.max(points, axis=0)
        bounding_box = Vector3()
        bounding_box.x = max_point[0] - min_point[0]  # size of the bounding box in x direction
        bounding_box.y = max_point[1] - min_point[1]  # size of the bounding box in y direction
        bounding_box.z = max_point[2] - min_point[2]  # size of the bounding box in z direction
        detection.bounding_box.size = bounding_box

        center = Pose()
        center.position.x = (min_point[0] + max_point[0]) / 2.0  # x position of the center of the bounding box
        center.position.y = (min_point[1] + max_point[1]) / 2.0  # y position of the center of the bounding box
        center.position.z = (min_point[2] + max_point[2]) / 2.0  # z position of the center of the bounding box
        detection.bounding_box.center = center

        # set the pose field based on the point cloud
        centroid = np.mean(points, axis=0)
        detection.pose.header.frame_id = pcl_msg.header.frame_id
        detection.pose.pose.position.x = centroid[0]  # x position of the object
        detection.pose.pose.position.y = centroid[1]  # y position of the object
        detection.pose.pose.position.z = centroid[2]  # z position of the object
        detection.pose.pose.orientation.w = 1.0  # quaternion representing the orientation of the object

        return detection


    

    """
    def callback_detect_3d(self, req):
        resp = Detect3DResponse()

        # get images
        img = self.image

        detections = self._get_detections(img)
        for d2d in detections.detections:
            d3d = Detection3D()
            d3d.header = d2d.header
            d3d.results = d2d.results

            # todo estimate bbox poses
            d3d.bbox.center.position.x = d2d.bbox.center.x / 640 - 0.5
            d3d.bbox.center.position.y = d2d.bbox.center.y / 480 - 0.5
            d3d.bbox.center.position.z = 1

            d3d.bbox.center.orientation.w = 1

            d3d.bbox.size.x = 0.1
            d3d.bbox.size.y = 0.1
            d3d.bbox.size.z = 0.1

            resp.detections.append(d3d)

        if len(detections.detections) > 0 and self.publish_detections:
            msg = Detection3DArray()
            msg.header = d3d.header
            msg.detections = resp.detections
            self.detections_pub.publish(msg)

        return resp
    """
        
    
        

    # Not my job...
    # Define callback function for clean object point cloud
    def pcl_2_gripper_pos(pcl_msg):
        # Convert point cloud message to PCL point cloud
        cloud = pcl_ros.point_cloud2_to_xyzrgb(pcl_msg)
        pcl_cloud = pcl.PointCloud(np.array(cloud['x'], dtype=np.float32),
                                    np.array(cloud['y'], dtype=np.float32),
                                    np.array(cloud['z'], dtype=np.float32))

        # Define gripper reach region
        gripper_radius = 0.05
        gripper_height = 0.1
        gripper_center = np.array([0, 0, 0.05]) # position of the gripper relative to the object centroid
        gripper_matrix = np.eye(4)
        gripper_matrix[0, 3] = gripper_center[0]
        gripper_matrix[1, 3] = gripper_center[1]
        gripper_matrix[2, 3] = gripper_center[2]

        # Filter point cloud to only include points within the gripper's reach
        reach_filter = pcl.CloudClip()
        reach_filter.setNearFarValues(0, gripper_height)
        reach_filter.setInputCloud(pcl_cloud)
        reach_filter.setClipFunction(gripper_matrix)
        cloud_filtered = pcl.PointCloud()
        reach_filter.filter(cloud_filtered)

        # Calculate centroid of filtered point cloud
        centroid = np.mean(cloud_filtered.to_array(), axis=0)[:3]

        # Calculate principal axis of filtered point cloud
        cov = np.cov(cloud_filtered.to_array()[:, :3], rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        principal_axis = eig_vecs[:, np.argmax(eig_vals)]

        # Calculate normal vector of the plane formed by the principal axis and the centroid
        normal_vector = np.cross(principal_axis, np.array([0, 0, 1]))
        normal_vector /= np.linalg.norm(normal_vector)

        # Calculate position of gripper
        gripper_position = centroid + gripper_center + (gripper_radius * normal_vector)

        # Calculate orientation of gripper
        gripper_orientation_matrix = np.eye(4)
        gripper_orientation_matrix[:3, 0]