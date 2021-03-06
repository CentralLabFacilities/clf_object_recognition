// ros
#include <ros/ros.h>

// msg
#include <vision_msgs/Detection3D.h>
#include <vision_msgs/Detection3DArray.h>
#include <vision_msgs/Detection2D.h>
#include <vision_msgs/Detection2DArray.h>
#include <vision_msgs/ObjectHypothesisWithPose.h>

// srv
#include <clf_object_recognition_msgs/Detect3D.h>
#include <clf_object_recognition_msgs/Detect2D.h>

// opencv
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "clf_object_recognition_merger/oobb_calculate.h"

// ros
ros::NodeHandle* nh;
ros::ServiceClient segmentationClient;
ros::ServiceClient detection2DClient;

// TODO: read as param
bool visualize = false;
double threshold = 0.3;  // threshold for matching boxes, TODO: find appropriate value

void fixBoxes(std::vector<vision_msgs::Detection3D>& detections3D)
{
  // ROS_INFO_STREAM("calculate oobbs");
  for (auto& detection : detections3D)
  {
    auto newbox = OOBB_Calculate::calculate(detection.source_cloud);
    // ROS_INFO_STREAM("BOX PRE: " << detection.bbox );
    // ROS_INFO_STREAM("BOX POST: " << newbox );
    detection.bbox = newbox;
  }
}

vision_msgs::BoundingBox2D bbox3Dto2D(const vision_msgs::BoundingBox3D bbox3D) {
    vision_msgs::BoundingBox2D estimatedBbox2D;
    // shift from origin from image center to upper left corner
    estimatedBbox2D.center.x = bbox3D.center.position.x / bbox3D.center.position.z + 0.5;
    estimatedBbox2D.center.y = bbox3D.center.position.y / bbox3D.center.position.z + 0.5;
    estimatedBbox2D.size_x = bbox3D.size.x / bbox3D.center.position.z;
    estimatedBbox2D.size_y = bbox3D.size.y / bbox3D.center.position.z;

    // estimated bboxes might be too large for the image: check and resize
    if ((estimatedBbox2D.center.x + estimatedBbox2D.size_x / 2) > 1.0)
    {
      // resize x dimension
      double x_max = estimatedBbox2D.center.x + estimatedBbox2D.size_x / 2;
      double diff = x_max - 1.0;
      estimatedBbox2D.size_x = estimatedBbox2D.size_x - diff;
      estimatedBbox2D.center.x = estimatedBbox2D.center.x - diff / 2;
    }
    if ((estimatedBbox2D.center.x - estimatedBbox2D.size_x / 2) < 0.0)
    {
      // resize x dimension
      double x_min = estimatedBbox2D.center.x - estimatedBbox2D.size_x / 2;
      double diff = -x_min;
      estimatedBbox2D.size_x = estimatedBbox2D.size_x - diff;
      estimatedBbox2D.center.x = estimatedBbox2D.center.x + diff / 2;
    }
    if ((estimatedBbox2D.center.y + estimatedBbox2D.size_y / 2) > 1.0)
    {
      // resize y dimension
      double y_max = estimatedBbox2D.center.y + estimatedBbox2D.size_y / 2;
      double diff = y_max - 1.0;
      estimatedBbox2D.size_y = estimatedBbox2D.size_y - diff;
      estimatedBbox2D.center.y = estimatedBbox2D.center.y - diff / 2;
    }
    if ((estimatedBbox2D.center.y - estimatedBbox2D.size_y / 2) < 0.0)
    {
      // resize y dimension
      double y_min = estimatedBbox2D.center.y - estimatedBbox2D.size_y / 2;
      double diff = -y_min;
      estimatedBbox2D.size_y = estimatedBbox2D.size_y - diff;
      estimatedBbox2D.center.y = estimatedBbox2D.center.y + diff / 2;
    }
    return estimatedBbox2D;
}

/*vision_msgs::BoundingBox2D bbox3Dto2Dv2(const vision_msgs::BoundingBox3D bbox3D) {
    vision_msgs::BoundingBox2D estimatedBbox2D;
    float xmin=cam_image.width, ymin=cam_image.height, xmax=0.0, ymax=0.0;
    //Iterate through all 8 edges of the bounding box, project them on the camera image, and find a 2D box that includes them all.
    for(int k=-1; k<=1; k+=2) {
        for(int l=-1; l<=1; l+=2) {
            for(int m=-1; m<=1; m+=2) {
                float imx, imy;
                tf2::Vector3 vec(k*bbox3D.size.x/2.0, l*bbox3D.size.y/2.0, m*bbox3D.size.z/2.0), vec2(bbox3D.center.position.x, bbox3D.center.position.y, bbox3D.center.position.z);
                tf2::Quaternion rot;
                tf2::fromMsg(bbox3D.center.orientation, rot);
                tf2::Transform trans(rot);
                vec=trans*vec;
                vec+=vec2;
                imx=535.0*vec.x()/vec.z()+319.0;//TODO get camera matrix from /xtion/rgb/camera_info-topic
                imy=535.0*vec.y()/vec.z()+253.0;
                if(imx<xmin) xmin=imx;
                if(imx>xmax) xmax=imx;
                if(imy<ymin) ymin=imy;
                if(imy>ymax) ymax=imy;
            }
        }
    }
    if(xmin<0.0) xmin=0.0;
    if(ymin<0.0) ymin=0.0;
    if(xmax>cam_image.width) xmax=cam_image.width;
    if(ymax>cam_image.height) ymax=cam_image.height;
    estimatedBbox2D.center.x=(xmin+xmax)/2.0;
    estimatedBbox2D.center.y=(ymin+ymax)/2.0;
    estimatedBbox2D.size_x=xmax-xmin;
    estimatedBbox2D.size_y=ymax-ymin;
    return estimatedBbox2D;
}*/

/* this service provides a list of detections with hypothesis (id, score), 3d boundingbox and pointcloud
   by merging 2d and 3d detections */
bool detectObjectsCallback(clf_object_recognition_msgs::Detect3D::Request& req,
                           clf_object_recognition_msgs::Detect3D::Response& res)
{
  // call segmentation (3d)
  clf_object_recognition_msgs::Detect3D segmentCall;
  segmentationClient.call(segmentCall);
  std::vector<vision_msgs::Detection3D> detections3D = segmentCall.response.detections;
  ROS_INFO_STREAM("segmented " << detections3D.size() << " objects");

  // call object detection (2d)
  clf_object_recognition_msgs::Detect2D detect2DCall;
  detection2DClient.call(detect2DCall);
  std::vector<vision_msgs::Detection2D> detections2D = detect2DCall.response.detections;
  ROS_INFO_STREAM("detected " << detections2D.size() << " objects in 2d");

  if (detections2D.size() == 0)
  {
    ROS_INFO_STREAM("no detections in 2d image, return 3d detections");

    fixBoxes(detections3D);
    res.detections = detections3D;

    return true;
  }
  if (detections3D.size() == 0)
  {
    ROS_INFO_STREAM("no object were segmented, convert 2d detections and return");
    for (int i = 0; i < detections2D.size(); i++)
    {
      vision_msgs::Detection3D detection;
      detection.results = detections2D.at(i).results;
      detections3D.push_back(detection);
    }
    res.detections = detections3D;
    return true;
  }

  double likelihood[detections3D.size()][detections2D.size()];
  std::vector<vision_msgs::BoundingBox2D> estimatedBboxes;

  // compare 3d and 2d bboxes and set likelihood
  for (int i = 0; i < detections3D.size(); i++)
  {
    vision_msgs::BoundingBox3D bbox3D = detections3D.at(i).bbox;  // in camera frame

    vision_msgs::BoundingBox2D estimatedBbox2D=bbox3Dto2D(bbox3D);

    // keep list for visualization
    estimatedBboxes.push_back(estimatedBbox2D);

    for (int j = 0; j < detections2D.size(); j++)
    {
      // compare bbox2D and estimatedBbox2D
      vision_msgs::BoundingBox2D bbox2D = detections2D.at(j).bbox;

      // likelihood is area of intersection / area of union
      double x1 = (bbox2D.center.x - bbox2D.size_x / 2.0);
      double y1 = (bbox2D.center.y - bbox2D.size_y / 2.0);
      double x2 = (bbox2D.center.x + bbox2D.size_x / 2.0);
      double y2 = (bbox2D.center.y + bbox2D.size_y / 2.0);

      double x1_est = (estimatedBbox2D.center.x - estimatedBbox2D.size_x / 2.0);
      double y1_est = (estimatedBbox2D.center.y - estimatedBbox2D.size_y / 2.0);
      double x2_est = (estimatedBbox2D.center.x + estimatedBbox2D.size_x / 2.0);
      double y2_est = (estimatedBbox2D.center.y + estimatedBbox2D.size_y / 2.0);

      double x_min = std::max(x1_est, x1);
      double y_min = std::max(y1_est, y1);
      double x_max = std::min(x2_est, x2);
      double y_max = std::min(y2_est, y2);

      double area_of_intersection = std::max((x_max - x_min), 0.0) * std::max((y_max - y_min), 0.0);
      double area_est_box = estimatedBbox2D.size_x * estimatedBbox2D.size_y;
      double area_box = bbox2D.size_x * bbox2D.size_y;
      double area_of_union =
          area_est_box + area_box - area_of_intersection;  // assume the detected and estimated box cannot be 0
      likelihood[i][j] = area_of_intersection / area_of_union;

      ROS_INFO_STREAM("(" << i << "," << j << ") likelihood " << likelihood[i][j]);
    }
  }

  // keep track of merged 2d detections
  std::vector<bool> mergedDetection2D;
  for (int j = 0; j < detections2D.size(); j++)
  {
    mergedDetection2D.push_back(false);
  }

  // check which 2d and 3d detections are matching
  for (int i = 0; i < detections3D.size(); i++)
  {
    double maxLikelihood = 0.0;
    int index = -1;
    for (int j = 0; j < detections2D.size(); j++)
    {
      if (likelihood[i][j] > maxLikelihood && likelihood[i][j] > threshold)
      {
        maxLikelihood = likelihood[i][j];
        index = j;
      }
    }
    if (index != -1)
    {
      ROS_INFO_STREAM("matching boxes for 3D no. " << i << " and 2D no. " << index);
      // set hypotheses of detection 3d with corresponding 2d detection
      detections3D.at(i).results = detections2D.at(index).results;
      // TODO: set pose and covariance?

      // remember that 2d detection was merged
      mergedDetection2D.at(index) = true;
    }
  }

  // add unmerged detections2d to detection3d list without pointcloud and 3d bbox
  for (int i = 0; i < mergedDetection2D.size(); i++)
  {
    if (!mergedDetection2D.at(i))
    {
      vision_msgs::Detection3D detection;
      detection.results = detections2D.at(i).results;
      detections3D.push_back(detection);
    }
  }

  // visualize segmented boxes in 2d image
  if (visualize)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      // assume all detections have the same source image
      cv_ptr = cv_bridge::toCvCopy(detections2D.at(0).source_img, sensor_msgs::image_encodings::BGR8);

      // draw 3d results (red)
      for (int i = 0; i < estimatedBboxes.size(); i++)
      {
        int width = cv_ptr->image.cols;
        int height = cv_ptr->image.rows;
        // estimated box in normalized coordinates
        int x_min = (estimatedBboxes.at(i).center.x - estimatedBboxes.at(i).size_x / 2) * width;
        int y_min = (estimatedBboxes.at(i).center.y - estimatedBboxes.at(i).size_y / 2) * height;
        int x_max = (estimatedBboxes.at(i).center.x + estimatedBboxes.at(i).size_x / 2) * width;
        int y_max = (estimatedBboxes.at(i).center.y + estimatedBboxes.at(i).size_y / 2) * height;
        cv::rectangle(cv_ptr->image, cv::Point(x_min, y_min), cv::Point(x_max, y_max), CV_RGB(255, 0, 0));
        cv::putText(cv_ptr->image, std::to_string(i), cv::Point(x_min, y_min), 0, 1.0, CV_RGB(255, 0, 0));
      }

      // draw 2d results (green)
      for (int i = 0; i < detections2D.size(); i++)
      {
        int width = cv_ptr->image.cols;
        int height = cv_ptr->image.rows;
        vision_msgs::BoundingBox2D bbox2D = detections2D.at(i).bbox;
        // estimated box in normalized coordinates where (0,0) is the left upper corner
        int x_min = (bbox2D.center.x - bbox2D.size_x / 2) * width;
        int y_min = (bbox2D.center.y - bbox2D.size_y / 2) * height;
        int x_max = (bbox2D.center.x + bbox2D.size_x / 2) * width;
        int y_max = (bbox2D.center.y + bbox2D.size_y / 2) * height;
        cv::rectangle(cv_ptr->image, cv::Point(x_min, y_min), cv::Point(x_max, y_max), CV_RGB(0, 255, 0));
        cv::putText(cv_ptr->image, std::to_string(i), cv::Point(x_min, y_min), 0, 1.0, CV_RGB(0, 255, 0));
      }

      cv::imshow("OPENCV_WINDOW", cv_ptr->image);
      cv::waitKey(0);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
    }
  }

  fixBoxes(detections3D);
  res.detections = detections3D;

  return true;
}

int main(int argc, char* argv[])
{
  ros::init(argc, argv, "object_merger");
  nh = new ros::NodeHandle("~");

  ros::ServiceServer objectDetectionService = nh->advertiseService("detect_objects", detectObjectsCallback);

  auto srv_seg = "/segment";
  auto srv_detect = "/detect";

  ROS_INFO_STREAM("waiting for service: " << srv_seg);
  ros::service::waitForService(srv_seg, -1);
  ROS_INFO_STREAM("waiting for service: " << srv_detect);
  ros::service::waitForService(srv_detect, -1);
  segmentationClient = nh->serviceClient<clf_object_recognition_msgs::Detect3D>(srv_seg);
  detection2DClient = nh->serviceClient<clf_object_recognition_msgs::Detect2D>(srv_detect);

  nh->getParam("show_image", visualize);
  nh->getParam("threshold", threshold);
  ROS_INFO_STREAM("show images: " << visualize);
  ROS_INFO_STREAM("threshold: " << threshold);

  ros::spin();
  return 0;
}
