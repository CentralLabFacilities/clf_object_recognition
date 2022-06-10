# ros
import rospy
import rostopic
import rosservice

#cv
from cv_bridge import CvBridge, CvBridgeError

#msgs
from vision_msgs.msg import ObjectHypothesis, Classification2D, Detection3D, Detection2D, Detection3DArray
from sensor_msgs.msg import Image

#srv
from clf_object_recognition_msgs.srv import Classify2D, Detect3D, Detect2D, Detect3DResponse


class SimpleDetect():

    def __init__(self, publish_detections):

        self.publish_detections = publish_detections

        # Bridge for opencv conversion
        self.bridge = CvBridge()

        self.sub_classify = rospy.ServiceProxy("/classify", Classify2D)
        self.sub_detect = rospy.ServiceProxy('/detect', Detect2D)
        #self.sub_images = rospy.Subscriber("~input", Image, self._image_callback)
        #rospy.logwarn("Listening to %s" % self.sub_images.name)

        self.pub = rospy.Publisher('/simple_detections', Detection3DArray, queue_size=10)

        self.service = rospy.Service("simple_detect", Detect3D, self.callback_detect_3d)

    def callback_detect_3d(self, req):
        try:
            rospy.wait_for_message("~input", Image, timeout=2)
        except rospy.ROSException as e:
            s = "could not get image from '"+rospy.resolve_name("~input")+"'"
            rospy.logerr(s)
            raise rospy.ServiceException(s)

        resp = Detect3DResponse()

        detections = self._get_detections(self._image)
        for d2d in detections:
            d3d = Detection3D()
            d3d.header = d2d.header
            # todo call classify for better hypotheses
            d3d.results = d2d.results
            # todo calc bbox real poses
            d3d.bbox.center.position.x = d2d.bbox.center.x 
            d3d.bbox.center.position.y = d2d.bbox.center.y 
            d3d.bbox.center.position.z = 1

            d3d.bbox.center.orientation.w = 1

            d3d.bbox.size.x = 0.1
            d3d.bbox.size.y = 0.1
            d3d.bbox.size.z = 0.1

            resp.detections.append(d3d)

        if self.publish_detections:
            self.pub.publish(resp.detections)

        return resp

    def _image_callback(self, msg):
        """
        Sensor_msgs/Image callback
        :param msg: The image message
        """
        self._image = msg

    def _get_classifications(self, images):
        try:
            result = self.sub_classify(images)
            return result
        except Exception as e:
            rospy.logerr("Service call failed: %s"%e)

    def _get_detections(self, image):
        try:
            result = self._srv_detect(image)
            return result
        except Exception as e:
            rospy.logerr("Service call failed: %s"%e)

#if __name__ == '__main__':
#    rospy.init_node('simple_detect')
#    publish_detections = rospy.get_param('publish_detections', True)
#    rospy.loginfo(("not " if not publish_detections else "")+"publishing detections")
#    SimpleDetect(publish_detections)
#    rospy.spin()