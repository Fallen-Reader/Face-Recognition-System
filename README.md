# Face Recognition Attendance System

A webcam-based face recognition system. It recognises enrolled people in a live video feed, marks them present in an attendance log, and has a small web page to enroll people, watch the camera and download the log.

## How it works

```
camera frame -> detector -> encoder -> matcher -> attendance log
                (find faces)  (128 numbers   (nearest enrolled
                               per face)      face within tolerance?)
```

1. **Detector** finds face boxes in the frame (`face_recognition`, HOG model).
2. **Encoder** turns each face into a 128-number vector (dlib's deep network).
3. **Matcher** compares it with every enrolled vector using Euclidean distance. The closest one wins if its distance is at most `MATCH_TOLERANCE` (0.5), otherwise the face is shown as **Unknown**.
4. **Attendance** logs a person once they have been recognised in 3 processed frames in a row, and at most once every 5 minutes.

## Project structure

```
Face_recog/
├── src/
│   ├── config.py      settings (paths, tolerance, camera, attendance)
│   ├── detector.py    find faces in a frame
│   ├── encoder.py     face -> 128-d encoding
│   ├── matcher.py     nearest-face matching
│   ├── database.py    SQLite storage for enrolled faces
│   ├── camera.py      plain OpenCV window (no web page)
│   ├── app.py         FastAPI web app: live feed, enroll, attendance
│   └── static/index.html
├── scripts/
│   ├── capture_faces.py   take photos of a person with the webcam
│   ├── enroll.py          encode the photos into the database
│   └── evaluate.py        accuracy report
├── data/              created automatically; photos + database (not in git)
├── requirements.txt
└── README.md
```

## Setup

You need Python 3.10 or newer and a webcam. Use a virtual environment.

**1. Install the build tools.** `dlib` compiles from source.

- Fedora: `sudo dnf install cmake gcc-c++ make python3-devel`
- Ubuntu/Debian: `sudo apt install cmake build-essential python3-dev`
- Windows: install CMake and the "Desktop development with C++" part of Visual Studio Build Tools.
- macOS: `brew install cmake`

**2. Install the packages.**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If compiling `dlib` keeps failing, skip it and install the prebuilt `dlib-bin` instead (same `import dlib`), then the other packages by hand:

```bash
pip install dlib-bin
pip install --no-deps face_recognition
pip install face_recognition_models Click Pillow numpy opencv-python "setuptools<82" fastapi "uvicorn[standard]"
```

pip may print a red "dependency conflict" line saying `dlib` is not installed. That is harmless here, because `dlib-bin` provides the same `dlib` module. Do not run `pip install -r requirements.txt` after this; it would try to compile `dlib` again.

**3. Check it works.**

```bash
python -c "import face_recognition; print('ok')"
```

## Usage

Run these from the project root, in order:

```bash
python scripts/capture.py   # asks for a name; press s to save a photo (3-5 per person), q to quit
python scripts/Enroll.py          # encodes the photos into the database
python src/app.py                 # starts the web app, then open http://127.0.0.1:8000
```

On the web page you can watch the live feed, enroll people with the camera (no photos needed), stop or start the camera, see and download the attendance log, and quit the app with the **Quit app** button.

Prefer a plain window? Run `python src/Camera.py` (press `q` to quit, `r` to reload the database).

### Settings (`src/config.py`)

| Setting | Default | Meaning |
|---|---|---|
| `MATCH_TOLERANCE` | 0.5 | Maximum distance for a match. Lower is stricter. |
| `PROCESS_EVERY_N_FRAMES` | 3 | Run recognition on every Nth frame. Raise it if the video lags. |
| `DETECTION_MODEL` | `"hog"` | `"hog"` is fast on CPU; `"cnn"` is more accurate but needs a GPU. |
| `ATTENDANCE_COOLDOWN` | 300 | Seconds before the same person can be logged again. |
| `ATTENDANCE_CONFIRM_FRAMES` | 3 | Processed frames in a row needed before logging. |

The last two are optional; add them to `config.py` to change them.

## Accuracy

```bash
python scripts/evaluate.py
```

It tests every enrolled person against the others and prints how often it recognises someone correctly, names the wrong person, rejects a real person, or accepts a stranger, at several tolerances. It also saves every trial to `data/eval_trials.csv`.

<!-- Paste your evaluate.py results here, and say how many people and photos you tested with. -->

## Limitations

- **No liveness check.** A photo or a video of an enrolled person shown to the camera can be accepted.
- **Image quality matters.** Low-resolution webcams, backlight and dim rooms lower accuracy. Enroll in the same kind of light you will use it in.
- **Small test group.** Results from a handful of people are only indicative, and photos taken in one sitting make them look better than real use.
- **Faces should be roughly frontal.** The HOG detector misses strongly turned or tilted faces.
- **Accuracy can differ between people.** The recognition model does not perform equally well for every skin tone, age or face shape, so test on the people who will actually use it.

## Privacy

Face photos and encodings are personal biometric data. They stay on your machine in `data/`, which is excluded from git. Only enroll people who agree to it, and keep the server on `127.0.0.1` (the default); it has no login.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Failed building wheel for dlib` | Install the build tools above, or use `pip install dlib-bin`. |
| `Please install face_recognition_models` | Usually a hidden `pkg_resources` error. Run `pip install "setuptools<82"`. |
| `compute_face_descriptor(): incompatible function arguments` | Convert frames with `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`, not `frame[:, :, ::-1]`. |
| Page says "Could not open the camera" | Another program is using the webcam. Close it, then click **Start camera**. |
| Enroll skips photos ("found 0 faces") | Face the light, move closer, keep the head straight. |
| A stranger gets a name | Lower `MATCH_TOLERANCE`, add more varied photos, run `evaluate.py`. |
| `Address already in use` | An old copy is still running. Close it, or change the port at the bottom of `app.py`. |

## Credits

Built on [`face_recognition`](https://github.com/ageitgey/face_recognition) by Adam Geitgey, [dlib](http://dlib.net/), OpenCV and FastAPI.
