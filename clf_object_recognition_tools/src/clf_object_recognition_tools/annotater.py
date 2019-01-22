import os
import rospy
import rostopic

import datetime

import cv2

from qt_gui.plugin import Plugin

from python_qt_binding.QtWidgets import *
from python_qt_binding.QtGui import *
from python_qt_binding.QtCore import *

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

import utils
import tf_utils

from image_widget import ImageWidget
from dialogs import warning_dialog


class AnnotationPlugin(Plugin):

    def __init__(self, context):
        """
        Annotation plugin to create and edit data sets either by manual annotation or automatically, e.g. using
        a tracker, and generating larger data sets with data augmentation.
        :param context: Parent QT widget
        """
        super(AnnotationPlugin, self).__init__(context)

        # Widget setup
        self.setObjectName('Label Plugin')

        self.widget = QWidget()
        context.add_widget(self.widget)
        self.widget.resize(800, 1000)

        """left side (current image, grab img button, ...)"""

        self.cur_im_widget = ImageWidget(self.widget, self.roi_callback, clear_on_click=False)
        self.cur_im_widget.setGeometry(QRect(20, 20, 640, 480))

        self.grab_img_button = QPushButton(self.widget)
        self.grab_img_button.setText("Grab frame")
        self.grab_img_button.clicked.connect(self.grab_frame)
        self.grab_img_button.setGeometry(QRect(20, 600, 100, 50))

        """right side (selected image, workspace...)"""

        self.sel_im_widget = ImageWidget(self.widget, self.roi_callback, clear_on_click=True)
        self.sel_im_widget.setGeometry(QRect(720, 20, 640, 480))

        """list widgets for images and annotations"""

        self.annotation_list_widget = QListWidget(self.widget)
        self.annotation_list_widget.setGeometry(QRect(1400, 50, 150, 200))
        self.annotation_list_widget.setObjectName("annotation_list_widget")
        self.annotation_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.annotation_list_widget.currentItemChanged.connect(self.select_annotation)

        self.image_list_widget = QListWidget(self.widget)
        self.image_list_widget.setGeometry(QRect(1550, 50, 250, 500))
        self.image_list_widget.setObjectName("image_list_widget")
        self.image_list_widget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.image_list_widget.currentItemChanged.connect(self.select_image)

        self.output_path_edit = QLineEdit(self.widget)
        self.output_path_edit.setGeometry(QRect(1400, 20, 300, 30))
        self.output_path_edit.setDisabled(True)

        self.edit_path_button = QPushButton(self.widget)
        self.edit_path_button.setText("set ws")
        self.edit_path_button.setGeometry(QRect(1700, 20, 100, 30))
        self.edit_path_button.clicked.connect(self.get_workspace)

        """ buttons for adding or deleting annotations"""
        self.add_annotation_button = QPushButton(self.widget)
        self.add_annotation_button.setText("add")
        self.add_annotation_button.setGeometry(QRect(1400, 250, 75, 30))
        self.add_annotation_button.clicked.connect(self.add_annotation)

        self.remove_annotation_button = QPushButton(self.widget)
        self.remove_annotation_button.setText("del")
        self.remove_annotation_button.setGeometry(QRect(1475, 250, 75, 30))
        self.remove_annotation_button.clicked.connect(self.remove_current_annotation)

        """label combo box, line edit and button for adding labels"""
        self.option_selector = QComboBox(self.widget)
        self.option_selector.currentIndexChanged.connect(self.class_change)
        self.option_selector.setGeometry(1400, 280, 150, 30)

        self.label_edit = QLineEdit(self.widget)
        self.label_edit.setGeometry(QRect(1400, 310, 100, 30))
        self.label_edit.setDisabled(False)

        self.edit_label_button = QPushButton(self.widget)
        self.edit_label_button.setText("add")
        self.edit_label_button.setGeometry(QRect(1500, 310, 50, 30))
        self.edit_label_button.clicked.connect(self.add_label)

        """ button for image deletion"""
        self.remove_image_button = QPushButton(self.widget)
        self.remove_image_button.setText("delete image")
        self.remove_image_button.setGeometry(QRect(1550, 550, 150, 30))
        self.remove_image_button.clicked.connect(self.remove_current_image)

        """ export data """
        self.export_ws_button = QPushButton(self.widget)
        self.export_ws_button.setText("Export workspace")
        self.export_ws_button.setGeometry(QRect(1550, 650, 150, 50))
        self.export_ws_button.clicked.connect(self.export_workspace_to_tf)

        """ functional stuff"""
        self.bridge = CvBridge()

        self.sub = None

        self.class_id = -1
        self.label = ""
        self.changes_done = False

        self.workspace = None
        self.labels = []
        self.images_with_annotations = []
        self.cur_annotated_image = None
        self.cur_annotation_index = -1

        self.class_change()

        self.default_config_path = None
        self.p_test = 0.2
        self.batch_size = 12

    def set_export_variables(self):
        """ Set variables for export via QInputDialog and QFileDialog. """
        # default config path
        config_path = QFileDialog.getOpenFileName(self.widget, "Select default config")
        config_path = str(config_path[0])
        file, ext = os.path.splitext(config_path)
        if not ext == ".config":
            self.default_config_path = None
            warning_dialog("warning", "invalid file extension")
            return
        else:
            self.default_config_path = config_path

        # batch size
        batch_size, ok = QInputDialog.getText(self.widget, "Set batch size", "12")
        if ok:
            try:
                self.batch_size = int(batch_size)
            except ValueError:
                pass

        # test percentage
        p_test, ok = QInputDialog.getText(self.widget, "Set test percentage", "0.2")
        if ok:
            try:
                self.p_test = float(p_test)
            except ValueError:
                pass

    def export_workspace_to_tf(self):
        """ Export workspace to training formats. """
        if self.default_config_path is None:
            warning_dialog("Warning", "define default config first")
            return

        tf_utils.export_data_to_tf(self.workspace, self.images_with_annotations, self.labels, self.p_test,
                                   self.default_config_path, self.batch_size, None, False)
        tf_utils.create_roi_images(self.workspace, self.images_with_annotations, self.labels)
        print("Export done")

    def add_label(self):
        """ If label doesn't exist yet, add to the list and combo box. """
        new_label = str(self.label_edit.text())
        if new_label is None or new_label == "":
            return

        for label in self.labels:
            if label[0] == new_label:
                warning_dialog("warning", "label\""+label[0]+"\" already exists")
                return
        new_label = list((new_label, 0))
        self.labels.append(new_label)
        self.option_selector.addItem(new_label[0])

        label_file = self.workspace+"/labels.txt"
        utils.write_labels(label_file, self.labels)

    def add_annotation(self):
        """ Add annotation to current image. Bounding box is just a dummy. Use current label. """
        if self.class_id == -1 or self.label == "":
            warning_dialog("Warning", "select label first")
            return
        if self.cur_annotated_image is None:
            warning_dialog("Warning", "select image first")
            return
        label = self.option_selector.currentText()
        annotation = utils.AnnotationWithBbox(self.class_id, 1.0, 0.5, 0.5, 1, 1)
        self.cur_annotated_image.annotation_list.append(annotation)
        index = len(self.cur_annotated_image.annotation_list)-1
        self.add_annotation_to_list_widget(index, label, True)

        self.changes_done = True

    def class_change(self):
        """ Called another label is selected in the combo box. Set current label and class id. """
        self.label = self.option_selector.currentText()
        if self.labels is None or len(self.labels) == 0:
            return
        self.class_id = [i[0] for i in self.labels].index(self.label)
        # num_annotations = self.labels[self.class_id][1]

    def select_annotation(self):
        """
            Called when an annotation is selected. The corresponding bounding box is drawn thicker,
            enables drawing on the image widget.
        """
        item = self.annotation_list_widget.currentItem()
        if item is None:
            self.sel_im_widget.set_active(False)
            self.sel_im_widget.clear()
            return
        text = str(item.text())
        index = text.split(":")[0]
        self.cur_annotation_index = int(index)
        self.show_image_from_workspace()
        self.sel_im_widget.set_active(True)

    def select_image(self):
        """
            Called when an image from the list is selected.
            Save annotations, if changes were made. Then show selected image and its annotation list.
        """
        if self.changes_done:
            utils.save_annotations(self.cur_annotated_image.image_file, self.cur_annotated_image.annotation_list)
            self.changes_done = False

        item = self.image_list_widget.currentItem()
        if item is None:
            return
        image_file = self.workspace + str(item.text())
        self.cur_annotated_image = None
        for img in self.images_with_annotations:
            if img.image_file == image_file:
                self.cur_annotated_image = img
        if self.cur_annotated_image is not None:
            self.show_image_from_workspace()
            self.set_annotation_list()

    def set_annotation_list(self):
        """ Set list of annotations for the current image. """
        self.annotation_list_widget.clear()
        self.cur_annotation_index = -1
        for i in range(len(self.cur_annotated_image.annotation_list)):

            index = int(self.cur_annotated_image.annotation_list[i].label)
            num_labels = len(self.labels)
            if index < num_labels:
                self.add_annotation_to_list_widget(i, self.labels[index][0])
            else:
                self.add_annotation_to_list_widget(i, "unknown")
        self.show_image_from_workspace()

    def add_annotation_to_list_widget(self, index, label, select=False):
        """
            Add a QListWidgetItem to the annotation QListWidget
            :param index: index in annotation list
            :param label: label of annotation
            :param select: select new item or not
        """
        item = QListWidgetItem()
        item.setText(str(index)+":"+label)
        self.annotation_list_widget.addItem(item)
        if select:
            self.annotation_list_widget.setCurrentItem(item)
            self.select_annotation()

    def add_image_to_list_widget(self, file_name, select=False):
        """
            Add a QListWidgetItem to the image QListWidget
            :param file_name: image file name
            :param select: select new item or not
        """
        item = QListWidgetItem()
        item.setText(file_name)
        self.image_list_widget.addItem(item)
        if select:
            self.image_list_widget.setCurrentItem(item)
            self.select_image()

    def refresh_image_list_widget(self):
        """ Clear and fill image list widget """
        self.image_list_widget.clear()
        for img in self.images_with_annotations:
            self.add_image_to_list_widget(img.image_file.replace(self.workspace, ""))

    def remove_current_annotation(self):
        """ Remove currently selected annotation """
        if self.cur_annotation_index == -1:
            warning_dialog("Warning", "no annotation selected")
            return
        del(self.cur_annotated_image.annotation_list[self.cur_annotation_index])
        self.set_annotation_list()

    def remove_current_image(self):
        """ Remove currently selected image. Remove ItemWidget and delete files """
        if self.cur_annotated_image is None:
            warning_dialog("Warning", "no image selected")
            return
        image_file = self.cur_annotated_image.image_file
        label_file = image_file.replace("images", "labels").replace("jpg", "txt")
        self.image_list_widget.removeItemWidget(self.image_list_widget.currentItem())
        self.images_with_annotations.remove(self.cur_annotated_image)

        os.remove(image_file)
        if os.path.isfile(label_file):
            os.remove(label_file)

        self.refresh_image_list_widget()

    def get_workspace(self):
        """ Gets and sets the output directory via a QFileDialog. Save before, if changes were done. """
        if self.changes_done:
            utils.save_annotations(self.cur_annotated_image.image_file, self.cur_annotated_image.annotation_list)
            self.changes_done = False

        self.load_workspace(QFileDialog.getExistingDirectory(self.widget, "Select output directory"))

    def load_workspace(self, path):
        """
        Sets the workspace directory. Checks for missing directories and files,
        then loads all images, annotations & label list.
        :param path: The path of the directory
        """
        if not path:
            path = "/tmp"

        self.workspace = path
        self.output_path_edit.setText(path)

        # clear all lists and references
        self.labels = []
        self.cur_annotation_index = -1
        self.cur_annotated_image = None
        self.images_with_annotations = []
        self.changes_done = False
        self.class_id = -1
        self.label = ""

        # clear combo box and listWidgets
        self.option_selector.clear()
        self.image_list_widget.clear()
        self.annotation_list_widget.clear()

        if utils.check_workspace(self.workspace):
            image_dir = self.workspace + "/images"
            label_file = self.workspace + "/labels.txt"

            self.labels = utils.read_labels(label_file)
            for label in self.labels:
                self.option_selector.addItem(label[0])

            for dirname, dirnames, filenames in os.walk(image_dir):
                for filename in filenames:
                    image_file = dirname + '/' + filename
                    label_file = image_file.replace("images", "labels").replace(".jpg", ".txt").replace(".png", ".txt")
                    annotated_image = utils.read_annotated_image(image_file, label_file)
                    self.images_with_annotations.append(annotated_image)
                    self.add_image_to_list_widget("/images/" + filename)
            print("workspace loaded successfully")

    def grab_frame(self):
        """ Grab current frame, save to workspace and show it."""
        if self.workspace is None or self.workspace == "":
            warning_dialog("Warning", "select workspace first")
            return
        cv_image = self.cur_im_widget.get_image()
        if cv_image is None:
            warning_dialog("warning", "no frame to grab")
            return
        file_name = "/images/{}.jpg".format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_%f"))
        new_image = utils.AnnotatedImage(self.workspace+file_name, [])
        self.cur_annotated_image = new_image
        self.images_with_annotations.append(new_image)
        cv2.imwrite(self.workspace+file_name, cv_image)
        self.add_image_to_list_widget(file_name, True)
        self.select_image()
#        self.sel_im_widget.set_image(cv_image, None, None)

    def roi_callback(self):
        """ Called when a roi is drawn on an image_widget. Edit annotation if possible. """
        if self.class_id == -1 or self.label == "":
            warning_dialog("Warning", "select label first")
            self.changes_done = True
            return
        if self.cur_annotation_index < 0:
            warning_dialog("Warning", "select annotation first")
            self.changes_done = True
            return

        # set bounding box
        x_center, y_center, width, height = self.sel_im_widget.get_normalized_roi()
        annotation = self.cur_annotated_image.annotation_list[self.cur_annotation_index]
        annotation.bbox = utils.BoundingBox(x_center, y_center, width, height)

        # set selected label
        self.label = self.option_selector.currentText()
        if self.labels is None or len(self.labels) == 0:
            return
        self.class_id = [i[0] for i in self.labels].index(self.label)
        annotation.label = self.class_id

        self.set_annotation_list()
        self.show_image_from_workspace()
        self.sel_im_widget.clear()

        self.changes_done = True

    def show_image_from_workspace(self):
        """ Show current selected image and annotations """
        if self.cur_annotated_image is None:
            return
        img = cv2.imread(self.cur_annotated_image.image_file)
        if img is not None:
            self.sel_im_widget.set_image(img, self.cur_annotated_image.annotation_list, self.cur_annotation_index)

    def image_callback(self, msg):
        """
        Called when a new sensor_msgs/Image is coming in
        :param msg: The image message
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        self.cur_im_widget.set_image(cv_image, None, None)

    def trigger_configuration(self):
        """
        Callback when the configuration button is clicked
        """
        topic_name, ok = QInputDialog.getItem(self.widget, "Select topic name", "Topic name",
                                              rostopic.find_by_type('sensor_msgs/Image'))
        if ok:
            self.create_subscriber(topic_name)

        self.set_export_variables()

    def create_subscriber(self, topic_name):
        """
        Method that creates a subscriber to a sensor_msgs/Image topic
        :param topic_name: The topic_name
        """
        if self.sub:
            self.sub.unregister()
        self.sub = rospy.Subscriber(topic_name, Image, self.image_callback)
        rospy.loginfo("Listening to %s -- spinning .." % self.sub.name)
        self.widget.setWindowTitle("Label plugin, listening to (%s)" % self.sub.name)

    def shutdown_plugin(self):
        """
        Callback function when shutdown is requested
        """
        if self.changes_done:
            utils.save_annotations(self.cur_annotated_image.image_file, self.cur_annotated_image.annotation_list)

    def save_settings(self, plugin_settings, instance_settings):
        """
        Callback function on shutdown to store the local plugin variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        instance_settings.set_value("workspace_dir", self.workspace)

    def restore_settings(self, plugin_settings, instance_settings):
        """
        Callback function fired on load of the plugin that allows to restore saved variables
        :param plugin_settings: Plugin settings
        :param instance_settings: Settings of this instance
        """
        workspace = None
        try:
            workspace = instance_settings.value("workspace_dir")
        except:
            pass
        self.load_workspace(workspace)
        self.create_subscriber(str(instance_settings.value("topic_name", "/xtion/rgb/image_raw")))