# clf_object_recognition_tensorflow

Ros object recognition based on tensorflow.

1. object_recognition_node (classification of objects)
    ```
    rosrun clf_object_recognition_tensorflow object_recognition_node _graph_path:=/path/to/recognition_dir/graph.pb _labels_path:=/path/to/recognition_dir/labels.txt
    ```

2. object_detection_node (detect objects in a 2D image, based on tensorflow detection api)
    ```
    rosrun clf_object_recognition_tensorflow object_detection_node _graph_path:=/path/to/graph.pb _labels_path:=/path/to/labelmap.pbtxt _rec_path:=/path/to/recognition_dir/
    ```
