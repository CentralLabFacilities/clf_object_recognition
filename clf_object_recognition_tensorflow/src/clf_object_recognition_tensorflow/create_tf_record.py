"""
Convert annotation files to tf record format
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
from PIL import Image

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

#TODO
from clf_object_recognition_tensorflow.objectset_utils import ObjectsetUtils

def create_tf_example(labelpath,imagepath,label_map,num_classes,util):
    with tf.gfile.GFile(os.path.join(imagepath), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = imagepath.encode('utf8')
    image_format = 'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # read label and roi coordinates from Yolo labels
    label = util.getLabelIdFromYolo(labelpath)
    x_min, y_min, x_max, y_max = util.getNormalizedRoiFromYolo(labelpath)

    # get class name from label map by id
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                     use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    id = int(label)+1
    class_text = category_index[id]['name']

    classes.append(id)
    classes_text.append(class_text.encode('utf8'))
    # use normalized coordinates
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


def create_config_and_records(input_path,output_path,default_config_path,fine_tune_checkpoint,batch_size):
    path = os.path.join(input_path)
    num_classes = 0
    # create label_map
    # TODO: handle if file is missing (just use directory names)
    print("expecting {}/classNames.txt".format(path))
    label_map_output = "{}/labelMap.pbtxt".format(output_path)
    class_name_path = "{}/classNames.txt".format(path)
    config_path = "{}/ssd.config".format(output_path)
    default_config_path = default_config_path
    # create output path if it doesnt exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(class_name_path) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        label_map_string = ""
        num_classes = len(content)
        for i in range(0, num_classes):
            label = content[i]
            label_map_string = label_map_string+"\nitem {\n  id:"+str(i+1)+"\n  name:\""+label+"\"\n}"
    with tf.gfile.Open(label_map_output, 'wb') as f:
        f.write(label_map_string)
    print('Successfully created the label map: {}'.format(label_map_output))

    # create tf record
    train_record_output = "{}/train.record".format(output_path)
    test_record_output = "{}/test.record".format(output_path)
    writer_train = tf.python_io.TFRecordWriter(train_record_output)
    writer_test = tf.python_io.TFRecordWriter(test_record_output)
    label_map = label_map_util.load_labelmap(label_map_output)

    util = ObjectsetUtils()
    train_test_counter = 0
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            labelpath = dirname + '/' + filename
            if 'labels' in labelpath and '.txt' in labelpath:
                imagepath = "{}/images/{}.jpg".format(dirname[:-7],filename[:-4])
                if (os.path.isfile(imagepath)):
                    print(imagepath)
                    tf_example = create_tf_example(labelpath,imagepath, label_map, num_classes,util)
                    if (tf_example == None):
                        continue
                    else:
                        if (train_test_counter < 7):
                            writer_train.write(tf_example.SerializeToString())
                        else:
                            writer_test.write(tf_example.SerializeToString())
                        train_test_counter = train_test_counter + 1
                        if (train_test_counter == 10):
                            train_test_counter = 0

    writer_train.close()
    writer_test.close()
    output_path = os.path.join(os.getcwd(), train_record_output)
    print('Successfully created the TFRecords: {}'.format(output_path))

    # create ssd default config
    default_config = open(default_config_path)
    new_config = open(config_path,'w')
    for line in default_config:
        newline = line
        if 'BATCH_SIZE' in line:
            newline = line.replace('BATCH_SIZE', str(batch_size))
        if 'FINE_TUNE_CHECKPOINT' in line:
            newline = line.replace('FINE_TUNE_CHECKPOINT', fine_tune_checkpoint)
        if 'FROM_DETECTION_CHECKPOINT' in line and fine_tune_checkpoint:
            newline = line.replace('FROM_DETECTION_CHECKPOINT', 'true')
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
        new_config.write(newline)

    return config_path