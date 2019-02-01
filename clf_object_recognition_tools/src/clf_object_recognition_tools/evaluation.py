# sys
import os
import datetime
import imghdr
import argparse


import cv2

# object rec
import utils
import tf_utils
from clf_object_recognition_tensorflow import detect
from clf_object_recognition_tensorflow import recognize


def match_hypotheses(num_hyp, hypotheses_list, ground_truth_id):
    """
    Compare the best n hypotheses with the ground truth label.
    :param num_hyp: number of hypotheses that shall be compared with the ground truth
    :param hypotheses_list: list of tuples (id, score)
    :param ground_truth_id: vision_msgs/ObjectHypothesisWithPose
    :return: likelihood of hypotheses matching
    """
    n = num_hyp
    if len(hypotheses_list) < num_hyp:
        n = len(hypotheses_list)
    print("Got {} hypotheses. Compare the best {} with ground truth ({})".format(len(hypotheses_list), n,
                                                                                 ground_truth_id))
    likelihood = 0.0
    score = 0
    for i in range(len(hypotheses_list)):
        if hypotheses_list[i][0] == ground_truth_id:
            if i < n:
                print("hypotheses {} matches".format(i))
                likelihood = 1.0/float(i+1)
            score = hypotheses_list[i][1]
    return likelihood, score


def match_bounding_boxes(detected_bbox, ground_truth):
    """
    Compare a detected bounding box with the ground truth box by intersection/union.
    :param detected_bbox: vision_msgs/BoundingBox2D
    :param ground_truth: vision_msgs/BoundingBox2D
    :return: likelihood of bounding boxes matching
    """
    detected_corners = detected_bbox.get_corners()
    ground_truth_corners = ground_truth.get_corners()
    x1 = max(detected_corners[0], ground_truth_corners[0])
    y1 = max(detected_corners[1], ground_truth_corners[1])
    x2 = min(detected_corners[2], ground_truth_corners[2])
    y2 = min(detected_corners[3], ground_truth_corners[3])

    # catch non-overlapping bounding boxes (use absolute values)
    width = x2-x1
    height = y2-y1

    if width < 0 or height < 0:
        return 0.0

    area_of_intersection = width*height
    if area_of_intersection < 0:
        print("ERROR")
    area_detected_bbox = detected_bbox.width*detected_bbox.height
    area_ground_truth_bbox = ground_truth.width * ground_truth.height
    area_of_union = area_detected_bbox+area_ground_truth_bbox-area_of_intersection

    likelihood = area_of_intersection / area_of_union

    return likelihood


def evaluate_detection(image_list, test_dict, graph_list, label_map, min_threshold=0.2, ignore_labels=False, do_recognition=False,
                       graph_r=None, labels_r=None, save_images=False, logging_dir=None):
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
        recognizer = recognize.TensorflowRecognition()
        recognizer.load_graph(graph_r, labels_r)

    for graph in graph_list:
        detector.load_graph(graph, label_map)
        # evaluate this graph and determine a good threshold
        for image in image_list:
            print(image)
            label_path = utils.get_label_path_by_image(image)
            annotation_list = utils.read_annotations(label_path)

            img = cv2.imread(image)
            id_list, score_list, box_list = detector.detect(img)
            detection_list = utils.detection_results_to_annotation_list(id_list, score_list, box_list)

            match_list = []
            for detection in detection_list:
                best_match = -1
                best_p = 0.2   # minimum threshold for matching
                label_match = False
                for i in range(len(annotation_list)):
                    annotation = annotation_list[i]
                    p = match_bounding_boxes(detection.bbox, annotation.bbox)
                    if ignore_labels:
                        if p > best_p:
                            best_match = i
                            best_p = p
                    elif not do_recognition:
                        if p > best_p:
                            hyp_list = [(detection.label, detection.prob)]
                            likelihood, score = match_hypotheses(1, hyp_list, annotation.label)
                            # todo: save threshold/score
                            if likelihood > 0:
                                label_match = True
                    else:
                        if p > best_p:
                            n = 3 #todo
                            hyp_list = [(detection.label, detection.prob)]  # todo: do recognition
                            likelihood, score = match_hypotheses(n, hyp_list, annotation.label)
                            # todo: save threshold/score
                            if likelihood > 0:
                                label_match = True

                match_list.append((best_match, best_p, label_match))

            if save_images:
                save_image(img, detection_list, annotation_list, match_list, logging_dir)


def save_image(image, detection_list, annotation_list, match_list, logging_dir):
    height, width, _ = image.shape

    green = (0, 255, 0)
    blue = (255, 0, 0)
    red = (0, 0, 255)
    for i in range(len(annotation_list)):
        draw_bounding_box(image, annotation_list[i], blue, width, height, i)
    for i in range(len(detection_list)):
        match_index = match_list[i][0]
        match_prob = match_list[i][1]
        if match_index == -1:
            draw_bounding_box(image, detection_list[i], red, width, height, match_prob)
        else:
            draw_bounding_box(image, detection_list[i], green, width, height, match_prob)

    filename = logging_dir + "eval_image_{}.jpg".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
    cv2.imwrite(filename, image)


def draw_bounding_box(image, annotation, color, width, height, prob):
    abs_box = utils.norm_to_abs_bbox(annotation.bbox, width, height)
    xmin, ymin, xmax, ymax = abs_box.get_corners()
    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
    box_label = '%s %f - %f' % (annotation.label, annotation.prob, prob)
    cv2.putText(image, box_label, (xmin, ymin), 0, 0.6, color, 1)


def evaluate_recognition(image_list, test_dict, rec_dict, graph_list, labels):
    """
    Evaluate list of recognition graphs. Either use a detection graph to get bounding boxes or have already filtered
    image regions in the image list. Threshold for detection and minimum threshold for recognition can be adjusted.
    A good threshold for recognition results will be found by evaluation.
    :param image_list: list of images to evaluate (either whole images (labels do exist) or rois)
    :param test_dict: dictionary of test set
    :param rec_dict: dictionary of recognition graph
    :param graph_list: list of recognition graph files
    :param labels: label file for recognition
    :return:
    """
    recognizer = recognize.TensorflowRecognition()
    hypotheses_to_compare = 3

    for graph in graph_list:
        print("load graph: "+graph)
        recognizer.load_graph(graph, labels)
        # evaluate this graph and determine a good threshold
        results = []

        for image in image_list:
            print(image)

            res = recognizer.recognize(image)
            res = list(reversed(res))
            print(res)  # sorted result (list of tupel: (id, prob)
            image_path_arr = image.split('/')
            label_name = image_path_arr[len(image_path_arr)-2]
            label_id = -1
            for key in rec_dict:
                if rec_dict[key] == label_name:
                    label_id = key
                    break
            if label_id == -1:
                print("WARNING: annotated roi has type \"{}\", that is not in the dictionary.".format(label_name))
            match = match_hypotheses(hypotheses_to_compare, res, label_id)
            print("match: "+str(match))
            results.append(match)

        sum_match = 0.0
        sum_threshold = 0.0
        sum_total = float(len(results))
        for res in results:
            sum_match = sum_match + res[0]
            sum_threshold = sum_threshold + res[1]

        recognition_rate = sum_match/sum_total
        average_threshold = sum_threshold/sum_total
        print("average threshold: " + str(average_threshold))
        print("recognition rate: " + str(recognition_rate))


def read_test_dir(test_dir, image_suffix, id_offset):
    """
    Scan image directory for all image files. If whole images, check that according label files exist.
    :param test_dir: directory of test set
    :param image_suffix: suffix for image directory (either /images or /rois)
    :param id_offset: offset for dictionary ids (depends on the format (txt file versus label map (.pbtxt))
    :return: list of image files
    """
    image_list = []
    image_dir = test_dir + image_suffix
    label_file = test_dir+"labels.txt"
    label_map_dict = None

    if os.path.isfile(label_file):
        label_map_dict = tf_utils.read_label_file(label_file, id_offset)

    for dirname, dirnames, filenames in os.walk(image_dir):
        for filename in filenames:
            image_path = dirname + '/' + filename
            label_path = utils.get_label_path_by_image(image_path)
            if (os.path.isfile(label_path) and imghdr.what(image_path)) or "/rois" in image_suffix:
                image_list.append(image_path)
    return image_list, label_map_dict


def get_detection_graphs_and_labels(graph_dir, label_map):
    """
    Scan graph dir for all available detection graphs and concatenate those to a list. Check label file exists.
    :param graph_dir: directory with one or multiple graphs
    :param label_map: path to label map
    :return: list of graphs and label map
    """
    label_dict = None
    graph_list = []

    if os.path.isfile(label_map):
        label_dict = tf_utils.read_label_map(label_map)

    for dirname, dirnames, filenames in os.walk(graph_dir):
        for filename in filenames:
            file_path = dirname + '/' + filename

            if "frozen_inference_graph.pb" in file_path:
                graph_list.append(file_path)

    return graph_list, label_dict


def get_recognition_graphs_and_labels(graph_dir, label_file):
    """
    Scan graph dir for all available recognition graphs and concatenate those to a list. Check label file exists.
    :param graph_dir: directory with one or multiple graphs
    :param label_file: path to label file
    :return: list of graphs and label file
    """
    label_dict = None
    graph_list = []

    if os.path.isfile(label_file):
        label_dict = tf_utils.read_label_file(label_file, 0)

    for dirname, dirnames, filenames in os.walk(graph_dir):
        for filename in filenames:
            file_path = dirname + '/' + filename

            if "output_graph.pb" in file_path:
                graph_list.append(file_path)

    return graph_list, label_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='evaluate.py: Use this script to evaluate tensorflow detection or recognition graphs. Results '
                    'including accuracy of specific graphs and threshold suggestions will be saved to a log file. '
                    'Displaying of results is possible. Select a mode (-m, default: 0) '
                    '\n0 - evaluate detection bounding boxes (ignore labels)'
                    '\n1 - evaluate detection (boxes + labels)'
                    '\n2 - evaluate detection with re-labeling'
                    '\n3 - evaluate recognition based on on predefined images')

    parser.add_argument('test_dir', type=str, help='sys-path to the directory with annotated test images')  # should be readable_dir
    parser.add_argument('logging_dir', type=str, help='sys-path to the directory where logs should be saved')  # should be readable_dir

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
    if mode < 0 or mode > 3:
        print("ERROR: Mode {} does not exist. Select one of the following modes:"
              "\n0 - evaluate detection bounding boxes (ignore labels)"
              "\n1 - evaluate detection (boxes + labels)"
              "\n2 - evaluate detection with re-labeling"
              "\n3 - evaluate recognition based on predefined images".format(mode))
        exit(1)

    # try to read graphs and label files, then check if all required are accessible
    detection_graphs, label_map_dict = get_detection_graphs_and_labels(detection_graph_dir, label_map_file)
    recognition_graphs, labels_dict = get_recognition_graphs_and_labels(recognition_graph_dir, recognition_labels)

    print("label map dict:")
    print(label_map_dict)

    print("label file dict:")
    print(labels_dict)

    if mode <= 2 and (not detection_graphs or label_map_dict is None):
        print("ERROR: Couldn't read detection graphs or label map.")
        exit(0)

    # get image list (either whole images or rois)
    image_list = []
    test_labels_dict = None
    if mode < 3:
        if mode < 2:
            image_list, test_labels_dict = read_test_dir(test_dir, "/images", 1)
        else:
            image_list, test_labels_dict = read_test_dir(test_dir, "/images", 0)
    else:
        image_list, test_labels_dict = read_test_dir(test_dir, "/rois", 0)    # mode 3

    if not image_list:
        print("ERROR: Couldn't find any valid images at "+test_dir)
        exit(0)

    if not test_labels_dict:
        print("ERROR: File does not exist or is empty: " + test_dir + "labels.txt")
        exit(0)

    print(test_labels_dict)

    if mode == 1:
        if test_labels_dict == label_map_dict:
            print("Label dictionaries match")
        else:
            print("Testset and trained graph cannot be compared, because they have different label dictionaries.")
            print("test: "+str(test_labels_dict))
            print("graph: "+str(label_map_dict))
            exit(0)

    if mode == 2:
        if len(test_labels_dict) == len(labels_dict):
            print("Label dictionaries match")
        else:
            # todo: merge somehow (ids of the graph as reference!)
            print("Testset and trained graph cannot be compared, because they have different label dictionaries.")
            print("test: "+str(test_labels_dict))
            print("graph: "+str(labels_dict))
            exit(0)

    # set threshold (default if argument is invalid)
    min_threshold = 0.2
    if threshold < 0 or threshold > 1:
        threshold = 0.5
    else:
        min_threshold = threshold

    # call evaluation according to selected mode
    if mode == 0:
        evaluate_detection(image_list, test_labels_dict, detection_graphs, label_map_file, min_threshold=min_threshold,
                           ignore_labels=True, do_recognition=False, graph_r=None, labels_r=None,
                           save_images=visualize, logging_dir=logging_dir)
    elif mode == 1:
        evaluate_detection(image_list, test_labels_dict, detection_graphs, label_map_file, min_threshold=min_threshold,
                           ignore_labels=False, do_recognition=False, graph_r=None, labels_r=None,
                           save_images=visualize, logging_dir=logging_dir)
    elif mode == 2:
        evaluate_detection(image_list, test_labels_dict, detection_graphs, label_map_file, min_threshold=min_threshold,
                           ignore_labels=True, do_recognition=True, graph_r=recognition_graphs[0],
                           labels_r=recognition_labels, save_images=visualize, logging_dir=logging_dir)
    elif mode == 3:
        evaluate_recognition(image_list, test_labels_dict, labels_dict, recognition_graphs, recognition_labels)

    print("Done")

