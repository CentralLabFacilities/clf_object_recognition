
class AnnotatedImage():
    """
    Init
    :param image_file: string with file name
    :param annotation_list: list of type AnnotationWithBbox
    """
    def __init__(self, image_file, annotation_list):
        self.image_file = image_file
        self.annotation_list = annotation_list

class AnnotationWithBbox():
    def __init__(self,label,prob,x_center,y_center,width,height):
        self.label = label
        self.prob = prob
        self.bbox = BoundingBox(x_center,y_center,width,height)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other): # python version perhaps cmp
        return self.__dict__ == other.__dict__

class BoundingBox():
    """
    Parameters are normalized image coordinates [0,1]
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

    def get_x_min(self):
        return self.x_center-self.width/2.0

    def get_y_min(self):
        return self.y_center-self.height/2.0

    def get_x_max(self):
        return self.x_center + self.width / 2.0

    def get_y_max(self):
        return self.y_center + self.height / 2.0
