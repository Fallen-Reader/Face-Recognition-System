import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWN_FACES_DIR = BASE_DIR / "data" / "known_faces"


def capture(name: str, camera_index: int = 0):
    person_dir = KNOWN_FACES_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    
    existing = list(person_dir.glob("img*.jpg"))
    count = len(existing)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    print("Press 's' to save a frame, 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        cv2.putText(frame, f"Saved: {count}  |  's'=save  'q'=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Capture - " + name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            count += 1
            out_path = person_dir / f"img{count}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"Saved {out_path}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    person_name = input("Enter name for this person: ").strip()
    capture(person_name)