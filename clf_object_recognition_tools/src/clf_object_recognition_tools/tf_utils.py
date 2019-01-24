import os
import shutil
import io
import random

import cv2

from PIL import Image
import tensorflow as tf
from object_detection.utils import dataset_util

import utils


def create_roi_images(ws_dir, annotated_images_list, label_list):
    """
    Create directory with roi images (content of bounding boxes) for a list of annotated images.
    :param ws_dir: workspace where rois are saved
    :param annotated_images_list: list of annotated images
    :param label_list: list with all labels
    """

    roi_dir = ws_dir+"/rois"
    if os.path.isdir(roi_dir):
        print("Delete existing dir: "+roi_dir)
        shutil.rmtree(roi_dir)  # delete older generated files

    os.makedirs(roi_dir)
    for label in label_list:
        os.makedirs(roi_dir+"/"+label[0])

    for annotated_image in annotated_images_list:

        image = cv2.imread(annotated_image.image_file)
        base_name = annotated_image.image_file.split("/")
        base_name = base_name[len(base_name)-1]

        h, w, c = image.shape

        for i in range(len(annotated_image.annotation_list)):
            a = annotated_image.annotation_list[i]
            label = label_list[int(a.label)][0]
            x_min, y_min, x_max, y_max = a.bbox.get_corners()
            roi = image[int(y_min*h):int(y_max*h), int(x_min*w):int(x_max*w)]
            h2, w2, c2 = roi.shape

            base_name = base_name.split('.')[0]
            output_file = roi_dir+"/"+label+"/"+base_name+"_"+str(i)+".jpg"
            if h2 == 0 or w2 == 0:
                print("skip roi with size 0: "+output_file)
                print(x_min, y_min, x_max, y_max)
            else:
                cv2.imwrite(output_file, roi)

    print("Successfully saved rois to: "+roi_dir)


def create_ssd_config(default_config_path, config_output, num_classes, label_map_output, test_record_output,
                      train_record_output, batch_size, checkpoint, use_checkpoint):
    """
    Create a ssd config based on a default config and replace some variables
    :param default_config_path: path to the default config
    :param config_output: output path for generated config
    :param num_classes: number of classes in total
    :param label_map_output: path to label map
    :param test_record_output: path to tf record with test set
    :param train_record_output: path to tf record with train set
    :param batch_size: batch size (training param)
    :param checkpoint: fine tune checkpoint (might be None)
    :param use_checkpoint: use checkpoint or not
    """
    default_config = open(default_config_path)
    new_config = open(config_output, 'w')
    for line in default_config:
        newline = line
        if 'NUM_CLASSES' in line:
            newline = line.replace('NUM_CLASSES', str(num_classes))
        if 'TEST_LABEL' in line:
            newline = line.replace('TEST_LABEL', label_map_output)
        if 'TRAIN_LABEL' in line:
            newline = line.replace('TRAIN_LABEL', label_map_output)
        if 'TEST_RECORD' in line:
            newline = line.replace('TEST_RECORD', test_record_output)
        if 'TRAIN_RECORD' in line:
            newline = line.replace('TRAIN_RECORD', train_record_output)
        if 'BATCH_SIZE' in line:
            newline = line.replace('BATCH_SIZE', str(batch_size))
        if 'FINE_TUNE_CHECKPOINT' in line and use_checkpoint:
            newline = line.replace('FINE_TUNE_CHECKPOINT', checkpoint)
        if 'FROM_DETECTION_CHECKPOINT' in line:
            newline = line.replace('FROM_DETECTION_CHECKPOINT', str(use_checkpoint).lower())
        new_config.write(newline)
    print("Sucessfully created config: {}".format(config_output))


def create_label_map(output_file, label_list):
    """
    Create a label map including id and name for all labels in the given list.
    :param output_file: path for output file
    :param label_list: list with all labels
    """
    label_map_string = ""
    for i in range(len(label_list)):
        name = label_list[i][0]
        # label map format starts with id 1
        label_map_string = label_map_string + "\nitem {\n  id:"+str(i+1)+"\n  name:\""+name+"\"\n}"
    with tf.gfile.Open(output_file, 'wb') as f:
        f.write(label_map_string)
    print('Successfully created the label map: {}'.format(output_file))


def export_data_to_tf(ws_dir, annotated_images_list, label_list, test_percentage, default_config, batch_size,
                      checkpoint, use_checkpoint):
    """
    Create config, label map and tf record files (train + test set) for the given workspace.
    :param ws_dir: workspace directory
    :param annotated_images_list: list of all annotated images in the workspace
    :param label_list: list of all labels
    :param test_percentage: percentage of test set [0,100]
    :param default_config: path to default config
    :param batch_size: batch size (write to config)
    :param checkpoint: fine tune checkpoint (write to config, might be None)
    :param use_checkpoint: use checkpoint or not
    """

    # create tf record
    train_record_output = ws_dir+"/train.record"
    test_record_output = ws_dir+"/test.record"
    label_map_output = ws_dir+"/label_map.pbtxt"
    config_output = ws_dir+"/ssd.config"
    writer_train = tf.python_io.TFRecordWriter(train_record_output)
    writer_test = tf.python_io.TFRecordWriter(test_record_output)

    counter_test = 0
    counter_train = 0
    test_percentage = int(test_percentage*100)

    for annotated_image in annotated_images_list:
        tf_example = create_tf_example(annotated_image, label_list)
        if random.randint(0, 100) < test_percentage:
            counter_test = counter_test + 1
            writer_test.write(tf_example.SerializeToString())
        else:
            counter_train = counter_train + 1
            writer_train.write(tf_example.SerializeToString())

    writer_train.close()
    writer_test.close()
    print("Write "+str(counter_train)+" train and "+str(counter_test)+" test examples")
    print('Successfully created the TFRecords: {}, {}'.format(train_record_output, test_record_output))

    # label map
    create_label_map(label_map_output, label_list)

    # config
    # todo: support other types of config
    create_ssd_config(default_config, config_output, len(label_list), label_map_output, test_record_output,
                      train_record_output, batch_size, checkpoint, use_checkpoint)


def create_tf_example(annotated_image, label_list):
    """
        Create tf example from AnnotatedImage object
        :param annotated_image: AnnotatedImage object
        :param label_list: list of all labels
    """
    image_file = annotated_image.image_file

    with tf.gfile.GFile(os.path.join(image_file), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = image_file.encode('utf8')
    image_format = 'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for a in annotated_image.annotation_list:
        class_id = int(a.label)+1  # tf record format starts with id 1
        class_text = label_list[int(a.label)][0]
        x_min, y_min, x_max, y_max = a.bbox.get_corners()
        classes_text.append(class_text)
        classes.append(class_id)
        xmins.append(x_min)
        xmaxs.append(x_max)
        ymins.append(y_min)
        ymaxs.append(y_max)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

