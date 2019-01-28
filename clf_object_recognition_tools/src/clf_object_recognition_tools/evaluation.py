# sys
import sys
import os
import copy
import imghdr
import argparse

# tensorflow
import tensorflow as tf

# object rec
import utils
import tf_utils
from clf_object_recognition_tensorflow import detect
from clf_object_recognition_tensorflow import recognize


# variables
cv_bridge = CvBridge()


def match_hypotheses(num_hyp, hypotheses_list, ground_truth):
    """
    Compare the best n hypotheses with the ground truth label.
    :param num_hyp: number of hypotheses that shall be compared with the ground truth
    :param hypotheses_list: vision_msgs/ObjectHypothesisWithPose[]
    :param ground_truth: vision_msgs/ObjectHypothesisWithPose
    :return: likelihood of hypotheses matching
    """
    n = num_hyp
    if len(hypotheses_list) < num_hyp:
        n = len(hypotheses_list)
    print("Got {} hypotheses. Compare the best {} with ground truth ({})".format(len(hypotheses_list), n,
                                                                                 ground_truth.id))
    likelihood = 0.0
    for i in range(n):
        if hypotheses_list[i].id == ground_truth.id:
            likelihood = 1.0/float(n)
    return likelihood


def match_bounding_boxes(detected_bbox, ground_truth):
    """
    Compare a detected bounding box with the ground truth box by intersection/union.
    :param detected_bbox: vision_msgs/BoundingBox2D
    :param ground_truth: vision_msgs/BoundingBox2D
    :return: likelihood of bounding boxes matching
    """
    x1 = max(detected_bbox.center.x-detected_bbox.size_x/2.0, ground_truth.center.x-ground_truth.size_x/2.0)
    y1 = max(detected_bbox.center.y-detected_bbox.size_y/2.0, ground_truth.center.y-ground_truth.size_y/2.0)
    x2 = min(detected_bbox.center.x+detected_bbox.size_x/2.0, ground_truth.center.x+ground_truth.size_x/2.0)
    y2 = min(detected_bbox.center.y+detected_bbox.size_y/2.0, ground_truth.center.x+ground_truth.size_y/2.0)

    area_of_intersection = (x2-x1)*(y2-y1)
    area_detected_bbox = detected_bbox.size_x*detected_bbox.size_y
    area_ground_truth_bbox = ground_truth.size_x * ground_truth.size_y
    area_of_union = area_detected_bbox+area_ground_truth_bbox-area_of_intersection

    likelihood = area_of_intersection / area_of_union
    return likelihood


def evaluate_detection(image_list, graph_list, label_map, min_threshold=0.2, ignore_labels=False, do_recognition=False,
                       graph_r=None, labels_r=None):
    """
    Evaluate a list of detection graphs given a minimum threshold. Analyse results to find a good threshold.
    Optionally use recognition graph to re-label results or ignore the labels completely (just compare bounding boxes).
    :param image_list: list of images (label files exist)
    :param graph_list: list of detection graph files
    :param label_map: label map for detection
    :param min_threshold: minimum threshold to filter detection results
    :param ignore_labels: if true, only compare bounding boxes
    :param do_recognition: if true, re-label detection results
    :param graph_r: graph file for recognition
    :param labels_r: label file for recognition
    :return:
    """
    detector = detect.Detector(detection_threshold=min_threshold)
    if do_recognition:
        recognizer = recognize.Recognizer()
        recognizer.load_graph(graph_r, labels_r)

    for graph in graph_list:
        detector.load_graph(graph, label_map)
        # evaluate this graph and determine a good threshold
        for image in image_list:
            print(image)


def evaluate_recognition(image_list, graph_list, labels, do_detection, detection_graph=None, label_map=None,
                         detection_threshold=0.5):
    """
    Evaluate list of recognition graphs. Either use a detection graph to get bounding boxes or have already filtered
    image regions in the image list. Threshold for detection and minimum threshold for recognition can be adjusted.
    A good threshold for recognition results will be found by evaluation.
    :param image_list: list of images to evaluate (either whole images (labels do exist) or rois)
    :param graph_list: list of recognition graph files
    :param labels: label file for recognition
    :param do_detection: if true: use detection graph, else: images are already rois
    :param detection_graph: detection graph file
    :param label_map: label map file
    :param detection_threshold: threshold for detection
    :return:
    """
    if do_detection:
        detector = detect.Detector(detection_threshold=detection_threshold)
        detector.load_graph(detection_graph, label_map)

    recognizer = recognize.Recognizer()

    for graph in graph_list:
        recognizer.load_graph(graph, labels)
        # evaluate this graph and determine a good threshold
        for image in image_list:
            print(image)


def get_test_images(image_dir):
    """
    Scan image directory for all image files. If whole images, check that according label files exist.
    :param image_dir: directory of images (either .../images or .../rois
    :return: list of image files
    """
    return []


def get_detection_graphs_and_labels(graph_dir, label_map):
    """
    Scan graph dir for all available detection graphs and concatenate those to a list. Check label file exists.
    :param graph_dir: directory with one or multiple graphs
    :param label_map: path to label map
    :return: list of graphs and label map
    """
    return [], None


def get_recognition_graphs_and_labels(graph_dir, label_map):
    """
    Scan graph dir for all available recognition graphs and concatenate those to a list. Check label file exists.
    :param graph_dir: directory with one or multiple graphs
    :param label_map: path to label file
    :return: list of graphs and label file
    """
    return [], None


if __name__ == "__main__":

    #example for argument parsing
    parser = argparse.ArgumentParser(
        description='evaluate.py: Use this script to evaluate tensorflow detection or recognition graphs. Results '
                    'including accuracy of specific graphs and threshold suggestions will be saved to a log file. '
                    'Displaying of results is possible. Select a mode (-m, default: 0) '
                    '\n0 - evaluate detection bounding boxes (ignore labels)'
                    '\n1 - evaluate detection (boxes + labels)'
                    '\n2 - evaluate detection with re-labeling'
                    '\n3 - evaluate recognition based on results of detection graph'
                    '\n4 - evaluate recognition based on predefined images')

    parser.add_argument('test_dir', type=readable_dir, help='sys-path to the directory with annotated test images')
    parser.add_argument('logging_dir', type=valid_dir, help='sys-path to the directory where logs should be saved')

    parser.add_argument('-m', '--mode', type=int, default=0, help='display images with results')
    parser.add_argument('-dg', '--detection_graphs', type=str, default="", help='sys-path to detection graph or '
                                                                                'multiple graphs')
    parser.add_argument('-dl', '--detection_labels', type=str, default="", help='sys-path to label map for detection')
    parser.add_argument('-rg', '--recognition_graphs', type=str, default="", help='sys-path recognition graph or '
                                                                                  'multiple graphs')
    parser.add_argument('-rl', '--recognition_labels', type=str, default="", help='sys-path recognition labels')
    parser.add_argument('-t', '--threshold', type=float, default=-1.0, help='(minimum) threshold for detection [0,1]')
    parser.add_argument('-v', '--visualize', type=bool, default=False, help='display images with results')

    args = parser.parse_args()

    test_dir = args.test_dir
    logging_dir = args.logging_dir

    mode = args.mode
    detection_graph_dir = args.detection_graphs
    label_map_file = args.detection_labels
    recognition_graph_dir = args.recognition_graphs
    recognition_labels = args.recognition_labels
    threshold = args.threshold
    visualize = args.visualize

    # check valid mode
    if mode < 0 or mode > 4:
        print("ERROR: Mode {} does not exist. Select one of the following modes:"
              "\n0 - evaluate detection bounding boxes (ignore labels)"
              "\n1 - evaluate detection (boxes + labels)"
              "\n2 - evaluate detection with re-labeling"
              "\n3 - evaluate recognition based on results of detection graph"
              "\n4 - evaluate recognition based on predefined images)".format(mode))
        exit(1)

    # try to read graphs and label files, then check if all required are accessible
    detection_graphs, label_map = get_detection_graphs_and_labels(detection_graph_dir, label_map_file)
    recognition_graphs, labels = get_recognition_graphs_and_labels(recognition_graph_dir, recognition_labels)

    if mode <= 2 and (not detection_graphs or label_map is None):
        print("ERROR: Couldn't read detection graphs or label map.")
        exit(0)

    if mode >= 2 and (not recognition_graphs or labels is None):
        print("ERROR: Couldn't read recognitions graphs or labels.")
        exit(0)

    #if mode == 1:
        # todo: check label map vs test labels
    #if mode >= 2:
        # todo: check rec labels vs test labels

    # get image list (either whole images or rois)
    image_list = []
    if mode < 4:
        image_list = get_test_images(test_dir + "/images")
    else:
        image_list = get_test_images(test_dir + "/rois")    # mode 4

    if not image_list:
        print("ERROR: Couldn't find any valid images at "+test_dir)
        exit(0)

    # set threshold (default if argument is invalid)
    min_threshold = 0.2
    if threshold < 0 or threshold > 1:
        threshold = 0.5
    else:
        min_threshold = threshold

    # call evaluation according to selected mode
    if mode == 0:
        evaluate_detection(image_list, detection_graphs, label_map, min_threshold=min_threshold, ignore_labels=True,
                           do_recognition=False, graph_r=None, labels_r=None)
    elif mode == 1:
        evaluate_detection(image_list, detection_graphs, label_map, min_threshold=min_threshold, ignore_labels=False,
                           do_recognition=False, graph_r=None, labels_r=None)
    elif mode == 2:
        evaluate_detection(image_list, detection_graphs, label_map, min_threshold=min_threshold, ignore_labels=True,
                           do_recognition=True, graph_r=recognition_graphs[0], labels_r=labels)
    elif mode == 3:
        evaluate_recognition(image_list, recognition_graphs, labels, do_detection=True,
                             detection_graph=detection_graphs[0], label_map=label_map, detection_threshold=0.5)
    elif mode == 4:
        evaluate_recognition(image_list, recognition_graphs, labels, do_detection=False, detection_graph=None,
                             label_map=None, detection_threshold=0.5)

    print("Done")

