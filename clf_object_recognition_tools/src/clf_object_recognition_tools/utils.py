import os
import cv2


def get_bbox_by_mask(mask):
    """ Returns bounding box in absolute image coordinates """
    version_number = int(cv2.__version__[0])  # check opencv version
    contours = 0
    if version_number > 2:
        mask, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    else:
        # this doesnt work for some reason (mask is just black)
        mask_cpy = mask.copy()
        contours, hierarchy = cv2.findContours(mask_cpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    largest_contour_index = 0
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], False)
        if a > largest_area:
            largest_area = a
            largest_contour_index = i
    box = cv2.boundingRect(contours[largest_contour_index])
    h, w = mask.shape
    center_x = float(box[0]) + float(box[2])/2.0
    center_y = float(box[1]) + float(box[3])/2.0
    width = box[2]
    height = box[3]
    bbox = BoundingBoxAbs(center_x, center_y, width, height, w, h)
    return bbox


def read_labels(file_name):
    """
        Read list of labels from file like this:
        label1
        label2
        label3
        ...
    """
    labels_tmp = []
    try:
        with open(file_name, 'r') as f:
            labels_tmp = f.readlines()
        labels_tmp = [(i.rstrip(), 0) for i in labels_tmp]
    except:
        pass

    labels = []
    for label in labels_tmp:
        label = list(label)
        labels.append(label)
    return labels


def write_labels(file_name, labels):
    """
        Write list of labels to file like this:
        label1
        label2
        label3
        ...
    """
    label_file = open(file_name, 'w')
    label_str = ""
    for label in labels:
        label_str = label_str + label[0] + "\n"
    label_file.write(label_str)
    label_file.close()


def detection_results_to_annotation_list(id_list, score_list, box_list):
    annotation_list = []

    for i in range(len(id_list)):
        label = id_list[i]
        prob = score_list[i]
        x_center = (box_list[i][1] + box_list[i][3])/2.0
        y_center = (box_list[i][0] + box_list[i][2])/2.0
        width = box_list[i][3] - box_list[i][1]
        height = box_list[i][2] - box_list[i][0]
        annotation = AnnotationWithBbox(label, prob, x_center, y_center, width, height)
        annotation_list.append(annotation)

    return annotation_list


def read_annotations(label_file):
    """
        Read annotation file of one image
        :return list of AnnotationWithBbox objects
    """
    annotation_list = []
    if os.path.isfile(label_file):
        with open(label_file) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
        for i in range(len(content)):
            line = content[i].split(' ')
            if len(line) == 5:
                annotation = AnnotationWithBbox(line[0], 1.0, line[1], line[2], line[3], line[4])
                annotation_list.append(annotation)
    else:
        print("label file missing: " + label_file)
    return annotation_list


def abs_to_norm_bbox(bbox_abs):
    x_center = float(bbox_abs.x_center)/float(bbox_abs.img_width)
    y_center = float(bbox_abs.y_center)/float(bbox_abs.img_height)
    width = float(bbox_abs.width)/float(bbox_abs.img_width)
    height = float(bbox_abs.height)/float(bbox_abs.img_height)

    return BoundingBox(x_center, y_center, width, height)


def norm_to_abs_bbox(bbox_norm, img_width, img_height):
    x_center = int(bbox_norm.x_center*img_width)
    y_center = int(bbox_norm.y_center*img_height)
    width = int(bbox_norm.width*img_width)
    height = int(bbox_norm.height*img_height)
    return BoundingBoxAbs(x_center, y_center, width, height, img_width, img_height)


def read_annotated_image(image_file, label_file):
    """ Read annotation file of one image return an AnnotatedImage object """
    return AnnotatedImage(image_file, read_annotations(label_file))


def save_annotations(image_file, annotation_list):
    """ Save annotations of one image. Intended to use with jpg or png images. """
    file_name = image_file.replace("images", "labels").replace("jpg", "txt").replace("png", "txt")
    label_str = ""
    for a in annotation_list:
        label_str = label_str + "{} {} {} {} {}\n".format(a.label, a.bbox.x_center, a.bbox.y_center, a.bbox.width,
                                                          a.bbox.height)
    print("write labels to: "+file_name)
    label_file = open(file_name, 'w')
    label_file.write(label_str)


def check_workspace(ws_dir):
    """
        Check current workspace for label list, images and annotation files.
        Create empty directories or label file, if they don't exist yet.
    """
    if ws_dir is None:
        return False

    print("checking workspace"+ws_dir)

    label_file = ws_dir + "/labels.txt"
    image_dir = ws_dir + "/images"
    label_dir = ws_dir + "/labels"

    if os.path.isdir(ws_dir):
        if not os.path.isfile(label_file):
            write_labels(label_file, [])
        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        if not os.path.isdir(label_dir):
            os.makedirs(label_dir)
    else:
        os.makedirs(ws_dir)
        os.makedirs(image_dir)
        os.makedirs(label_dir)
        write_labels(label_file, [])
    return True


def get_label_path_by_image(image_path):
    label_path = image_path.replace("/images/", "/labels/").replace("jpg", "txt").replace("png", "txt")
    return label_path


def get_jpg_by_label_path(label_path):
    image_path = label_path.replace("/labels/", "/images/").replace("txt", "jpg")
    return image_path


class AnnotatedImage:
    def __init__(self, image_file, annotation_list):
        """
            :param image_file: string with file name
            :param annotation_list: list of type AnnotationWithBbox
        """
        self.image_file = image_file
        self.annotation_list = annotation_list


class AnnotationWithBbox:
    def __init__(self, label, prob, x_center, y_center, width, height):
        """
        :param label: id of annotation
        :param prob: probability of annotation [0,1]
        :param x_center: x coordinate of bbox center [0,1]
        :param y_center: y coordinate of bbox center [0,1]
        :param width: width of bbox [0,1]
        :param height: height of bbox [0,1]
        """
        self.label = label
        self.prob = prob
        self.bbox = BoundingBox(x_center, y_center, width, height)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class BoundingBoxAbs:
    """
        Store absolute image coordinates of a bounding box and the image dimensions.
    """
    def __init__(self, x_center, y_center, width, height, img_width, img_height):
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height
        self.img_width = img_width
        self.img_height = img_height

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_corners(self):
        return self.get_x_min(), self.get_y_min(), self.get_x_max(), self.get_y_max()

    def get_x_min(self):
        return self.x_center - self.width / 2

    def get_y_min(self):
        return self.y_center - self.height / 2

    def get_x_max(self):
        return self.x_center + self.width / 2

    def get_y_max(self):
        return self.y_center + self.height / 2


class BoundingBox:
    """
        Store normalized image coordinates [0,1] of a bounding box.
    """
    def __init__(self, x_center, y_center, width, height):
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)
        self.height = float(height)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def get_corners(self):
        return self.get_x_min(), self.get_y_min(), self.get_x_max(), self.get_y_max()

    def get_x_min(self):
        return self.x_center-self.width / 2.0

    def get_y_min(self):
        return self.y_center-self.height / 2.0

    def get_x_max(self):
        return self.x_center + self.width / 2.0

    def get_y_max(self):
        return self.y_center + self.height / 2.0
