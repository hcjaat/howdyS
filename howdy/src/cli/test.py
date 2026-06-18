import configparser
import builtins
import os
import json
import sys
import time
import cv2
import numpy as np
import paths_factory
from models.recognition import get_face_embedding, match_faces
from i18n import _
from recorders.video_capture import VideoCapture

config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

video_capture = VideoCapture(config)

similarity_threshold = config.getfloat("video", "certainty", fallback=7.5) / 10
exposure = config.getint("video", "exposure", fallback=-1)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)

print(_("""
Opening a window with a test feed

Press ctrl+C in this terminal to quit
Click on the image to enable or disable slow mode
"""))


def mouse(event, x, y, flags, param):
    global slow_mode
    if event == cv2.EVENT_LBUTTONDOWN:
        slow_mode = not slow_mode


def print_text(line_number, text):
    cv2.putText(
        overlay, text, (10, height - 10 - (10 * line_number)),
        cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0), 0, cv2.LINE_AA,
    )


encodings = []
models = None

try:
    user = builtins.howdy_user
    models = json.load(open(paths_factory.user_model_path(user)))
    for model in models:
        encodings += model["data"]
except FileNotFoundError:
    pass

cv2.namedWindow("Howdy Test")
cv2.setMouseCallback("Howdy Test", mouse)

slow_mode = False
total_frames = 0
sec_frames = 0
fps = 0
sec = int(time.time())
rec_tm = 0

try:
    while True:
        frame_tm = time.time()
        total_frames += 1
        sec_frames += 1

        if sec != int(frame_tm):
            fps = sec_frames
            sec = int(frame_tm)
            sec_frames = 0

        frame = video_capture.read()
        if frame is None:
            continue

        gsframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gsframe = clahe.apply(gsframe)
        overlay = gsframe.copy()
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

        height, width = gsframe.shape[:2]

        hist = cv2.calcHist([gsframe], [0], None, [8], [0, 256])
        hist_total = int(sum(hist)[0])
        hist_perc = []

        for index, value in enumerate(hist):
            value_perc = float(value[0]) / hist_total * 100 if hist_total > 0 else 0
            hist_perc.append(value_perc)
            p1 = (20 + (10 * index), 10)
            p2 = (10 + (10 * index), int(value_perc / 2 + 10))
            cv2.rectangle(overlay, p1, p2, (0, 200, 0), thickness=cv2.FILLED)

        print_text(0, _("RESOLUTION: %dx%d") % (height, width))
        print_text(1, _("FPS: %d") % (fps,))
        print_text(2, _("FRAMES: %d") % (total_frames,))
        print_text(3, _("RECOGNITION: %dms") % (round(rec_tm * 1000),))

        if slow_mode:
            cv2.putText(
                overlay, _("SLOW MODE"), (width - 66, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 0, cv2.LINE_AA,
            )

        if hist_perc[0] > dark_threshold:
            cv2.putText(
                overlay, _("DARK FRAME"), (width - 68, 16),
                cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 0, cv2.LINE_AA,
            )
        else:
            cv2.putText(
                overlay, _("SCAN FRAME"), (width - 68, 16),
                cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0), 0, cv2.LINE_AA,
            )

            rec_tm = time.time()
            face_encoding = get_face_embedding(frame)
            rec_tm = time.time() - rec_tm

            if face_encoding is not None:
                color = (0, 0, 230)

                if models:
                    for i, saved_enc in enumerate(encodings):
                        saved = np.array(saved_enc)
                        similarity = np.dot(saved, face_encoding) / (
                            np.linalg.norm(saved) * np.linalg.norm(face_encoding)
                        )
                        similarity = float(similarity)

                        if similarity > similarity_threshold:
                            color = (0, 230, 0)
                            circle_text = "{} (similarity: {:.3f})".format(
                                models[i]["label"], similarity,
                            )
                            cv2.putText(
                                overlay, circle_text, (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 0), 0,
                                cv2.LINE_AA,
                            )
                        else:
                            cv2.putText(
                                overlay, "no match", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 0, 255), 0,
                                cv2.LINE_AA,
                            )

                cv2.putText(
                    overlay, _("FACE DETECTED"), (width - 90, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, .3, color, 0, cv2.LINE_AA,
                )

        alpha = 0.65
        frame_rgb = cv2.cvtColor(gsframe, cv2.COLOR_GRAY2BGR)
        cv2.addWeighted(overlay, alpha, frame_rgb, 1 - alpha, 0, frame_rgb)

        cv2.imshow("Howdy Test", frame_rgb)

        if cv2.waitKey(1) != -1:
            raise KeyboardInterrupt()

        frame_time = time.time() - frame_tm
        if slow_mode:
            time.sleep(max([.5 - frame_time, 0.0]))

        if exposure != -1:
            video_capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
            video_capture.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

except KeyboardInterrupt:
    print(_("\nClosing window"))
    cv2.destroyAllWindows()
