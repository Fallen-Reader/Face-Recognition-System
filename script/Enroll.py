import sys
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from Detector import FaceDetector 
from Encoder import FaceEncoder    
from Database import FaceDatabase  

KNOWN_FACES_DIR = BASE_DIR / "data" / "known_faces"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def enroll_all():
    if not KNOWN_FACES_DIR.exists():
        print(f"No folder at {KNOWN_FACES_DIR}. Run scripts/capture.py first.")
        return

    Detector, encoder, db = FaceDetector(), FaceEncoder(), FaceDatabase()

    for person_dir in sorted(p for p in KNOWN_FACES_DIR.iterdir() if p.is_dir()):
        name = person_dir.name
        encodings = []

        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  [skip] {img_path.name}: could not read image")
                continue

            locations = Detector.detect_faces(frame)
            if len(locations) != 1:
                print(f"  [skip] {name}/{img_path.name}: found {len(locations)} faces (need exactly 1)")
                continue

            encodings.extend(encoder.encode_faces(frame, locations))

        if not encodings:
            print(f"{name}: no usable photos, nothing enrolled")
            continue

        db.delete_face(name) 
        for enc in encodings:
            db.add_face(name, enc)
        print(f"{name}: enrolled {len(encodings)} encodings")

    print("\nIn database now:", db.list_people())


if __name__ == "__main__":
    enroll_all()