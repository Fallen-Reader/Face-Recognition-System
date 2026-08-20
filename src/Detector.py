import cv2 as cv
import face_recognition
from config import DETECTION_MODEL

class FaceDetector:
    def __init__(self, model: str = DETECTION_MODEL):
        self.model = model
 
    def detect_faces(self, frame):
        rgb_frame = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame, model=self.model)
        return face_locations

if __name__ == "__main__":
    face = FaceDetector()

    face.detect_faces()
