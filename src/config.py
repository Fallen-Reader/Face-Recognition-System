from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
Model_Path = BASE_DIR/"model"

#Camera
CAMERA_INDEX = 0            
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
PROCESS_EVERY_N_FRAMES = 3 

#model
DETECTION_MODEL = "hog"

#tolerance
MATCH_TOLERANCE = 0.6