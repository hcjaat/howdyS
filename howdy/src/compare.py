import time

timings = {"st": time.time()}

import sys
import os
import json
import configparser
from datetime import timezone, datetime
import atexit
import subprocess
import snapshot
import numpy as np
import cv2
import paths_factory
from models.recognition import get_face_embedding, match_faces
from recorders.video_capture import VideoCapture
from i18n import _


def exit(code=None):
    global gtk_proc
    if "gtk_proc" in globals():
        gtk_proc.terminate()
    if code is not None:
        sys.exit(code)


def make_snapshot(type):
    snapshot.generate(
        snapframes,
        [
            type + _(" LOGIN"),
            _("Date: ")
            + datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC"),
            _("Scan time: ") + str(round(time.time() - timings["fr"], 2)) + "s",
            _("Frames: ")
            + str(frames)
            + " ("
            + str(round(frames / (time.time() - timings["fr"]), 2))
            + "FPS)",
            _("Hostname: ") + os.uname().nodename,
            _("Best certainty value: ") + str(round(lowest_certainty, 3)),
        ],
    )


def send_to_ui(type, message):
    global gtk_proc
    if "gtk_proc" in globals():
        message = type + "=" + message + " \n"
        try:
            if gtk_proc.poll() is None:
                gtk_proc.stdin.write(bytearray(message.encode("utf-8")))
                gtk_proc.stdin.flush()
        except IOError:
            pass


if len(sys.argv) < 2:
    exit(12)

user = sys.argv[1]
models = []
encodings = []
black_tries = 0
dark_tries = 0
frames = 0
snapframes = []
lowest_certainty = 1.0

try:
    models = json.load(open(paths_factory.user_model_path(user)))
    for model in models:
        encodings += model["data"]
except FileNotFoundError:
    exit(10)

if len(models) < 1:
    exit(10)

config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

timeout = config.getint("video", "timeout", fallback=4)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=50.0)
similarity_threshold = config.getfloat("video", "certainty", fallback=7.5) / 10
end_report = config.getboolean("debug", "end_report", fallback=False)
save_failed = config.getboolean("snapshots", "save_failed", fallback=False)
save_successful = config.getboolean("snapshots", "save_successful", fallback=False)
gtk_stdout = config.getboolean("debug", "gtk_stdout", fallback=False)
rotate = config.getint("video", "rotate", fallback=0)
exposure = config.getint("video", "exposure", fallback=-1)

gtk_pipe = sys.stdout if gtk_stdout else subprocess.DEVNULL

try:
    gtk_proc = subprocess.Popen(
        ["howdy-gtk", "--start-auth-ui"],
        stdin=subprocess.PIPE,
        stdout=gtk_pipe,
        stderr=gtk_pipe,
    )
    atexit.register(exit)
except FileNotFoundError:
    pass

send_to_ui("M", _("Starting up..."))

timings["in"] = time.time() - timings["st"]

timings["ic"] = time.time()
video_capture = VideoCapture(config)
timings["ic"] = time.time() - timings["ic"]
timings["ll"] = 0

max_height = config.getfloat("video", "max_height", fallback=320.0)

send_to_ui("M", _("Identifying you..."))

frames = 0
valid_frames = 0
timings["fr"] = time.time()
dark_running_total = 0

encodings_arr = np.array(encodings) if encodings else np.array([])

while True:
    frames += 1

    ui_subtext = "Scanned " + str(valid_frames - dark_tries) + " frames"
    if dark_tries > 1:
        ui_subtext += " (skipped " + str(dark_tries) + " dark frames)"
    send_to_ui("S", ui_subtext)

    if time.time() - timings["fr"] > timeout:
        if save_failed:
            make_snapshot(_("FAILED"))
        if dark_tries == valid_frames:
            print(
                _("All frames were too dark, please check dark_threshold in config")
            )
            print(
                _("Average darkness: {avg}, Threshold: {threshold}").format(
                    avg=str(dark_running_total / max(1, valid_frames)),
                    threshold=str(dark_threshold),
                )
            )
            exit(13)
        else:
            exit(11)

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
        height = frame.shape[0]
        scaling_factor = (max_height / height) or 1

    if scaling_factor != 1:
        frame = cv2.resize(
            frame,
            None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )
        gsframe = cv2.resize(
            gsframe,
            None,
            fx=scaling_factor,
            fy=scaling_factor,
            interpolation=cv2.INTER_AREA,
        )

    if rotate == 1:
        if frames % 3 == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if frames % 3 == 2:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 2:
        if frames % 2 == 0:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            gsframe = cv2.rotate(gsframe, cv2.ROTATE_90_CLOCKWISE)

    face_encoding = get_face_embedding(frame)
    if face_encoding is None:
        continue

    for i, enc in enumerate(encodings):
        saved = np.array(enc)
        similarity = np.dot(saved, face_encoding) / (
            np.linalg.norm(saved) * np.linalg.norm(face_encoding)
        )
        similarity = float(similarity)

        if similarity < lowest_certainty:
            lowest_certainty = similarity

        if similarity > similarity_threshold:
            timings["tt"] = time.time() - timings["st"]
            timings["fl"] = time.time() - timings["fr"]

            if end_report:
                def print_timing(label, k):
                    print("  %s: %dms" % (label, round(timings[k] * 1000)))

                print(_("Time spent"))
                print_timing(_("Starting up"), "in")
                print(
                    _("  Open cam: %dms")
                    % (round(max(timings["ll"], timings["ic"]) * 1000)),
                )
                print_timing(_("  Opening the camera"), "ic")
                print_timing(_("Searching for known face"), "fl")
                print_timing(_("Total time"), "tt")

                print(_("\nResolution"))
                print(
                    _("  Native: %dx%d") % (frame.shape[1], frame.shape[0])
                )
                scale_height, scale_width = frame.shape[:2]
                print(
                    _("  Used: %dx%d") % (scale_height, scale_width)
                )

                print(
                    _("\nFrames searched: %d (%.2f fps)")
                    % (frames, frames / timings["fl"]),
                )
                print(_("Black frames ignored: %d ") % (black_tries,))
                print(_("Dark frames ignored: %d ") % (dark_tries,))
                print(
                    _("Similarity of winning frame: %.3f")
                    % (similarity,)
                )
                print(
                    _("Winning model: %d (\"%s\")")
                    % (i, models[i]["label"]),
                )

            if save_successful:
                make_snapshot(_("SUCCESSFUL"))

            if config.getboolean("rubberstamps", "enabled", fallback=False):
                import rubberstamps

                send_to_ui("S", "")
                if "gtk_proc" not in vars():
                    gtk_proc = None
                rubberstamps.execute(config, gtk_proc, {
                    "video_capture": video_capture,
                    "clahe": clahe,
                })

            exit(0)

    if exposure != -1:
        video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        video_capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
