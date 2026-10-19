from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "faces.db"
ENCODINGS_DIR = BASE_DIR / "data" / "encodings"

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 3

DETECTION_MODEL = "hog"

MATCH_TOLERANCE = 0.6