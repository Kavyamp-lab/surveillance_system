# detector.py
import cv2

class SubDetector:
    def __init__(self):
        # Using Haar cascades for ultra-fast CPU-bound face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_face(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)

        # Crop person region
        person_roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        # If no face detected
        if len(faces) == 0:
            return None

        # Take first detected face
        fx, fy, fw, fh = faces[0]

        # Convert back to full frame coordinates
        return (
            x1 + fx,
            y1 + fy,
            x1 + fx + fw,
            y1 + fy + fh
        )