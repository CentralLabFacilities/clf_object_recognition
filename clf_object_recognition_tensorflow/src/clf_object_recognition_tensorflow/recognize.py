
import tensorflow as tf
import numpy as np


class TensorflowRecognition:

    def __init__(self):
        self.labels = []
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True #If true, the allocator does not pre-allocate the entire specified GPU memory region, instead starting small and growing as needed.
        config.gpu_options.per_process_gpu_memory_fraction=0.25 #Don't use so much GPU memory!
        self.sess=tf.Session(config=config) #If the recognition is used very rarely, it might make sense to only open a session when it is needed and close it afterwards (with a context manager, "with tf.Session ...")


    def load_graph(self, graph_path, label_path):
        with open(graph_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
            with open(label_path, 'rb') as f:
                self.labels = f.read().split("\n")
        return self.labels


    def recognize(self,filename):
        """1. Get result tensor"""
        result_tensor = self.sess.graph.get_tensor_by_name("final_result:0")

        """2. Open Image and perform prediction"""
        predictions = []

        #TODO: read image from memory
        with open(filename, 'rb') as f: #TODO instead of reading the image from a file, pass it as a numpy-array parameter to recognize(), see here https://stackoverflow.com/questions/40273109/convert-python-opencv-mat-image-to-tensorflow-image-data and here https://stackoverflow.com/questions/34484148/feeding-image-data-in-tensorflow-for-transfer-learning Problem: rgb or bgr?
            predictions = self.sess.run(result_tensor, {'DecodeJpeg/contents:0': f.read()})
            predictions = np.squeeze(predictions)

        """3. Construct list with labels and probabilities"""
        result = zip(list(range(len(predictions))), predictions)

        # return sorted list
        sorted_result = sorted(result, key=lambda x: x[1])
        return sorted_result
