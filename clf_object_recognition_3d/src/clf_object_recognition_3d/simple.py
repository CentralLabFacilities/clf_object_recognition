# ros
import rospy

#msgs
from vision_msgs.msg import  Detection3D, Detection3DArray

#srv
from clf_object_recognition_msgs.srv import Detect3D, Detect2D, Detect3DResponse


class SimpleDetect():

    def __init__(self, detect2d_topic, publish_detections = True):

        self.publish_detections = publish_detections
        self.srv_detect = rospy.ServiceProxy(detect2d_topic, Detect2D)

        if publish_detections:
            self.pub = rospy.Publisher('/simple_detections', Detection3DArray, queue_size=10)

        self.service = rospy.Service("simple_detect", Detect3D, self.callback_detect_3d)

    def callback_detect_3d(self, req):
        resp = Detect3DResponse()

        detections = self._get_detections()
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
            self.pub.publish(msg)

        return resp

    def _get_detections(self):
        try:
            result = self.srv_detect()
            return result
        except Exception as e:
            raise rospy.ServiceException(e)
