import face_recognition

class FaceEncoder:
    def encode_faces(self, frame, face_locations):
        rgb_frame = frame[:, :, ::-1]
        face_encodings = face_recognition.face_encodings(rgb_frame, known_face_locations=face_locations)
        return face_encodings