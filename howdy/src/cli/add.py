import time
import os
import sys
import json
import configparser
import builtins
import numpy as np
import cv2
import paths_factory
from models.recognition import encode_face
from recorders.video_capture import VideoCapture
from i18n import _

user = builtins.howdy_user
enc_file = paths_factory.user_model_path(user)
encodings = []

if not os.path.exists(paths_factory.user_models_dir_path()):
    print(_("No face model folder found, creating one"))
    os.makedirs(paths_factory.user_models_dir_path())

try:
    encodings = json.load(open(enc_file))
except FileNotFoundError:
    encodings = []

if len(encodings) > 3:
    print(_("NOTICE: Each additional model slows down the face recognition engine slightly"))
    print(_("Press Ctrl+C to cancel\n"))

if not builtins.howdy_args.plain:
    print(_("Adding face model for the user ") + user)

label = "Initial model"
next_id = encodings[-1]["id"] + 1 if encodings else 0

if builtins.howdy_args.arguments:
    label = builtins.howdy_args.arguments[0]
else:
    label = _("Model #") + str(next_id)

if builtins.howdy_args.y:
    print(_('Using default label "%s" because of -y flag') % (label,))
else:
    label_in = input(_("Enter a label for this new model [{}]: ").format(label))
    if label_in != "":
        label = label_in[:24]

if "," in label:
    print(_('NOTICE: Removing illegal character "," from model name'))
    label = label.replace(",", "")

insert_model = {
    "time": int(time.time()),
    "label": label,
    "id": next_id,
    "data": [],
}

config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

video_capture = VideoCapture(config)

print(_("\nPlease look straight into the camera"))
time.sleep(2)

frames = 0
valid_frames = 0
dark_tries = 0
dark_running_total = 0
face_encoding = None

dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)

while frames < 60:
    frames += 1
    frame = video_capture.read()
    if frame is None:
        continue

    gsframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gsframe = clahe.apply(gsframe)

    hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
    hist_total = np.sum(hist)
    darkness = hist[0] / hist_total * 100 if hist_total > 0 else 100

    if hist_total == 0 or darkness == 100:
        continue

    dark_running_total += darkness
    valid_frames += 1

    if darkness > dark_threshold:
        dark_tries += 1
        continue

    embedding = encode_face(frame)
    if embedding is not None:
        face_encoding = embedding
        break

video_capture.release()

if face_encoding is None:
    if valid_frames == 0:
        print(_("Camera saw only black frames - is IR emitter working?"))
    elif valid_frames == dark_tries:
        print(_("All frames were too dark, please check dark_threshold in config"))
        print(
            _("Average darkness: {avg}, Threshold: {threshold}").format(
                avg=str(dark_running_total / valid_frames),
                threshold=str(dark_threshold),
            )
        )
    else:
        print(_("No face detected, aborting"))
    sys.exit(1)

insert_model["data"].append(face_encoding.tolist())
encodings.append(insert_model)

with open(enc_file, "w") as datafile:
    json.dump(encodings, datafile)

print(_("""\nScan complete\nAdded a new model to """) + user)
