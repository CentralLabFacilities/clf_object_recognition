from ultralytics import YOLO

class Recognizer(object):
    def __init__(self, model_file):
        self.model = YOLO(model_file)

    def inference(self, img, confthreshold=0.25):
        results = self.model.predict(
            img,
            conf=confthreshold,
            verbose=False
        )
        

        if len(results) == 0 or len(results[0].boxes) == 0:
            return ([], [], [], [])

        boxes = results[0].boxes

        cls = boxes.cls.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        bboxes = boxes.xyxy.cpu().numpy()

        masks = None
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()

        return cls, scores, bboxes, masks
