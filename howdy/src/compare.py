import time
import sys
import os
import json
import configparser
from datetime import timezone, datetime
import atexit
import subprocess
import numpy as np
import cv2
import onnxruntime as ort
import paths_factory
import snapshot
from recorders.video_capture import VideoCapture
from i18n import _

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DETECTION_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
RECOGNITION_MODEL = os.path.join(MODELS_DIR, "face_recognition_mobilefacenet.onnx")

_REC_SESSION = None
_HAAR_CASCADE = None


def _detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    global _HAAR_CASCADE
    if _HAAR_CASCADE is None:
        path = os.path.join(os.path.dirname(cv2.__file__), "data",
                            "haarcascade_frontalface_default.xml")
        if not os.path.exists(path):
            path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
        _HAAR_CASCADE = cv2.CascadeClassifier(path)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _HAAR_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
    return [(x, y, x + w, y + h) for x, y, w, h in faces]


def encode_face(frame: np.ndarray) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None

    faces = _detect_faces(frame)
    if not faces:
        return None

    x1, y1, x2, y2 = faces[0]
    if x2 <= x1 or y2 <= y1:
        return None

    face_crop = frame[y1:y2, x1:x2]
    face_blob = cv2.dnn.blobFromImage(
        face_crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )

    global _REC_SESSION
    if _REC_SESSION is None:
        _REC_SESSION = ort.InferenceSession(
            RECOGNITION_MODEL, providers=["CPUExecutionProvider"]
        )
    input_name = _REC_SESSION.get_inputs()[0].name
    embedding = _REC_SESSION.run(None, {input_name: face_blob})[0]
    return embedding.flatten()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def authenticate(user: str) -> bool:
    timings = {"st": time.time()}

    try:
        models = json.load(open(paths_factory.user_model_path(user)))
    except FileNotFoundError:
        return False

    if len(models) < 1:
        return False

    encodings = []
    for model in models:
        encodings += model["data"]

    timings["in"] = time.time() - timings["st"]

    config = configparser.ConfigParser()
    config.read(paths_factory.config_file_path())

    timeout = config.getint("video", "timeout", fallback=4)
    dark_threshold = config.getfloat("video", "dark_threshold", fallback=50.0)
    similarity_threshold = config.getfloat("video", "certainty", fallback=7.5) / 10
    save_failed = config.getboolean("snapshots", "save_failed", fallback=False)
    save_successful = config.getboolean("snapshots", "save_successful", fallback=False)
    rotate = config.getint("video", "rotate", fallback=0)
    exposure = config.getint("video", "exposure", fallback=-1)
    max_height = config.getfloat("video", "max_height", fallback=320.0)

    gtk_proc = None
    try:
        gtk_stdout = config.getboolean("debug", "gtk_stdout", fallback=False)
        gtk_pipe = sys.stdout if gtk_stdout else subprocess.DEVNULL
        gtk_proc = subprocess.Popen(
            ["howdy-gtk", "--start-auth-ui"],
            stdin=subprocess.PIPE, stdout=gtk_pipe, stderr=gtk_pipe,
        )

        def cleanup():
            if gtk_proc and gtk_proc.poll() is None:
                gtk_proc.terminate()
        atexit.register(cleanup)
    except FileNotFoundError:
        pass

    def send_to_ui(msg_type, message):
        if gtk_proc and gtk_proc.poll() is None:
            try:
                gtk_proc.stdin.write(
                    bytearray(f"{msg_type}={message} \n".encode("utf-8"))
                )
                gtk_proc.stdin.flush()
            except IOError:
                pass

    send_to_ui("M", _("Starting up..."))

    timings["ic"] = time.time()
    video_capture = VideoCapture(config)
    timings["ic"] = time.time() - timings["ic"]

    scaling_factor = 1.0
    frames = 0
    valid_frames = 0
    timings["fr"] = time.time()
    dark_running_total = 0
    black_tries = 0
    dark_tries = 0
    snapframes = []
    lowest_certainty = 1.0

    send_to_ui("M", _("Identifying you..."))

    while True:
        frames += 1

        ui_subtext = f"Scanned {valid_frames - dark_tries} frames"
        if dark_tries > 1:
            ui_subtext += f" (skipped {dark_tries} dark frames)"
        send_to_ui("S", ui_subtext)

        if time.time() - timings["fr"] > timeout:
            if save_failed:
                snapshot.generate(snapframes, [
                    _("FAILED LOGIN"),
                    _("Date: ") + datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC"),
                    _("Hostname: ") + os.uname().nodename,
                ])
            if dark_tries == valid_frames:
                print(_("All frames were too dark, please check dark_threshold in config"))
                print(_("Average darkness: {avg}, Threshold: {threshold}").format(
                    avg=str(dark_running_total / max(1, valid_frames)),
                    threshold=str(dark_threshold),
                ))
            return False

        frame = video_capture.read()
        if frame is None:
            continue

        gsframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gsframe = clahe.apply(gsframe)

        if save_failed or save_successful:
            if len(snapframes) < 3:
                snapframes.append(frame)

        hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
        hist_total = np.sum(hist)
        darkness = hist[0] / hist_total * 100 if hist_total > 0 else 100

        if hist_total == 0 or darkness == 100:
            black_tries += 1
            continue

        dark_running_total += darkness
        valid_frames += 1

        if darkness > dark_threshold:
            dark_tries += 1
            continue

        if frames == 1:
            scaling_factor = (max_height / frame.shape[0]) or 1

        if scaling_factor != 1:
            frame = cv2.resize(
                frame, None, fx=scaling_factor, fy=scaling_factor,
                interpolation=cv2.INTER_AREA,
            )

        if rotate == 1:
            if frames % 3 == 1:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if frames % 3 == 2:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate == 2:
            if frames % 2 == 0:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        face_encoding = encode_face(frame)
        if face_encoding is None:
            continue

        for i, enc in enumerate(encodings):
            similarity = cosine_similarity(np.array(enc), face_encoding)

            if similarity < lowest_certainty:
                lowest_certainty = similarity

            if similarity > similarity_threshold:
                timings["tt"] = time.time() - timings["st"]
                timings["fl"] = time.time() - timings["fr"]

                if save_successful:
                    snapshot.generate(snapframes, [
                        _("SUCCESSFUL LOGIN"),
                        _("Date: ") + datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC"),
                        _("Scan time: ") + f"{round(timings['fl'], 2)}s",
                        _("Similarity: ") + f"{similarity:.3f}",
                        _("Hostname: ") + os.uname().nodename,
                    ])

                if config.getboolean("rubberstamps", "enabled", fallback=False):
                    import rubberstamps
                    send_to_ui("S", "")
                    if gtk_proc is None:
                        gtk_proc = None
                    rubberstamps.execute(config, gtk_proc, {
                        "video_capture": video_capture,
                        "clahe": clahe,
                    })

                return True

        if exposure != -1:
            video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            video_capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(12)
    if authenticate(sys.argv[1]):
        sys.exit(0)
    sys.exit(11)
