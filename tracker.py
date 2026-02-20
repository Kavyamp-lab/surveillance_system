import cv2
from ultralytics import YOLO
from config import Config


class ObjectTracker:
    def __init__(self):
        self.model = YOLO(Config.YOLO_MODEL)

    def process_frame(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            conf=Config.CONF_THRESHOLD,
            classes=[0],  # Detect only 'person'
            verbose=False
        )

        tracked_objects = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()

                for box, track_id, conf in zip(boxes, ids, confs):
                    x1, y1, x2, y2 = map(int, box)

                    tracked_objects.append(
                        (x1, y1, x2, y2, int(track_id))
                    )

        return tracked_objects