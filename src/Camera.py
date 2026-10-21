import cv2
from config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, PROCESS_EVERY_N_FRAMES
from Detector import FaceDetector
from Encoder import FaceEncoder
from Matcher import FaceMatcher
from Database import FaceDatabase

SCALE = 0.25 


class CameraStream:
    def __init__(self):
        self.Detector = FaceDetector()
        self.Encoder = FaceEncoder()
        self.Matcher = FaceMatcher()
        self.db = FaceDatabase()
        self.reload_db()

    def reload_db(self):
        self.known_names, self.known_encodings = self.db.load_all()
        print(f"Loaded {len(self.known_names)} encodings from database.")

    def process_frame(self, frame):
        small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
        locations = self.Detector.detect_faces(small)
        encodings = self.Encoder.encode_faces(small, locations)

        results = []
        inv = int(1 / SCALE)
        for (top, right, bottom, left), enc in zip(locations, encodings):
            name, dist = self.Matcher.match(enc, self.known_encodings, self.known_names)
            results.append(((top * inv, right * inv, bottom * inv, left * inv), name, dist))
        return results

    @staticmethod
    def draw(frame, results):
        for (top, right, bottom, left), name, dist in results:
            known = name != "Unknown"
            color = (0, 200, 0) if known else (0, 0, 255)
            label = f"{name} ({dist:.2f})" if dist is not None else name
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def run(self):
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            print("Could not open camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        frame_count = 0
        results = []  

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                results = self.process_frame(frame)

            self.draw(frame, results)
            cv2.imshow("Face Recognition", frame)
            frame_count += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.reload_db()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CameraStream().run()