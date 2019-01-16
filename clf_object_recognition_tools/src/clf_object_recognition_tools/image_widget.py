from python_qt_binding.QtWidgets import * 
from python_qt_binding.QtGui import * 
from python_qt_binding.QtCore import * 
import cv2
import numpy as np
import copy

def _convert_cv_to_qt_image(cv_image):
    """
    Method to convert an opencv image to a QT image
    :param cv_image: The opencv image
    :return: The QT Image
    """
    cv_image = cv_image.copy() # Create a copy
    height, width, byte_value = cv_image.shape
    byte_value = byte_value * width
    cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)
    return QImage(cv_image, width, height, byte_value, QImage.Format_RGB888)


def _get_roi_from_rect(rect):
    """
    Returns the ROI from a rectangle, the rectangle can have the top and bottom flipped
    :param rect: Rect to get roi from
    :return: x, y, width, height of ROI
    """
    x_min = min(rect.topLeft().x(), rect.bottomRight().x())
    y_min = min(rect.topLeft().y(), rect.bottomRight().y())
    x_max = max(rect.topLeft().x(), rect.bottomRight().x())
    y_max = max(rect.topLeft().y(), rect.bottomRight().y())

    return x_min, y_min, x_max - x_min, y_max - y_min



class ImageWidget(QWidget):

    def __init__(self, parent, image_callback, clear_on_click=False):
        """
        Image widget that allows drawing rectangles and firing a image_roi_callback
        :param parent: The parent QT Widget
        :param image_callback: The callback function when a ROI is drawn
        """
        super(ImageWidget, self).__init__(parent)
        self._cv_image = None
        self._bg_image = None
        self._qt_image = QImage()

        self.clip_rect = QRect(0, 0, 0, 0)

        self._active = False
        self.dragging = False
        self.drag_offset = QPoint()
        self.image_callback = image_callback

        self.detections = []  # todo
        self._clear_on_click = clear_on_click

        self.bbox = None


    def paintEvent(self, event):
        """
        Called every tick, paint event of QT
        :param event: Paint event of QT
        """
        painter = QPainter()
        painter.begin(self)
        painter.drawImage(0, 0, self._qt_image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.cyan, 5.0))
        painter.drawRect(self.clip_rect)

        painter.setFont(QFont('Decorative', 10))
        for rect, label in self.detections:
            painter.setPen(QPen(Qt.magenta, 5.0))
            painter.drawRect(rect)

            painter.setPen(QPen(Qt.magenta, 5.0))
            painter.drawText(rect, Qt.AlignCenter, label)

        painter.end()

    def get_image(self):
        return self._cv_image

    def get_roi_image(self):
        # Flip if we have dragged the other way
        x, y, width, height = _get_roi_from_rect(self.clip_rect)
        self.bbox = (x, x+width, y, y+height)

        return self._cv_image[y:y + height, x:x + width]

    def set_image(self, img, annotation_list, sel_index):
        self._cv_image = copy.copy(img)
        if annotation_list is not None:
            for i in range(len(annotation_list)):
                a = annotation_list[i]

                color = (0, 0, 255)
                line_thickness = 1
                if i == sel_index:
                    line_thickness = 3

                height, width, channels = img.shape
                cv2.rectangle(img, (int(a.bbox.get_x_min()*width), int(a.bbox.get_y_min()*height)),
                              (int(a.bbox.get_x_max()*width), int(a.bbox.get_y_max()*height)),
                              color, line_thickness)
                cv2.putText(img, str(i), (int(a.bbox.get_x_min()*width), int(a.bbox.get_y_min()*height)), 0, 0.6, color, 1)
        self._qt_image = _convert_cv_to_qt_image(img)
        self.update()

    def set_active(self, active):
        self._active = active

    def clear(self):
        self.clip_rect = QRect(0, 0, 0, 0)

    def get_roi(self):
        return _get_roi_from_rect(self.clip_rect)

    def get_normalized_roi(self):
        x_min, y_min, width, height = self.get_roi()
        h, w, channels = self._cv_image.shape
        x_center = (float(x_min) + float(width) / 2.0) / float(w)
        y_center = (float(y_min) + float(height) / 2.0) / float(h)
        width = float(width) / float(w)
        height = float(height) / float(h)
        return x_center, y_center, width, height

    def mousePressEvent(self, event):
        """
        Mouspress callback
        :param event: mouse event
        """
        # Check if we clicked on the img
        if event.pos().x() < self._qt_image.width() and event.pos().y() < self._qt_image.height():
            # check if clicking is allowed
            if self._active:
                if self._clear_on_click:
                    self.clear()
                self.clip_rect.setTopLeft(event.pos())
                self.clip_rect.setBottomRight(event.pos())
                self.dragging = True

    def mouseMoveEvent(self, event):
        """
        Mousemove event
        :param event: mouse event
        """
        if not self.dragging:
            return

        self.clip_rect.setBottomRight(event.pos())
        
        self.update()
    
    def mouseReleaseEvent(self, event):
        """
        Mouse release event
        :param event: mouse event
        """
        if not self.dragging:
            return

        self.image_callback()

        self.dragging = False
