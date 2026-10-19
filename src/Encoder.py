import face_recognition
import cv2
class FaceEncoder:
    def encode_faces(self, frame, face_locations):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
        return face_encodings