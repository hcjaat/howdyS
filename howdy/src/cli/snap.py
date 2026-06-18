import os
import configparser
from datetime import timezone, datetime
import snapshot
import paths_factory
from recorders.video_capture import VideoCapture
from i18n import _

config = configparser.ConfigParser()
config.read(paths_factory.config_file_path())

video_capture = VideoCapture(config)

video_capture.read()

exposure = config.getint("video", "exposure", fallback=-1)
dark_threshold = config.getfloat("video", "dark_threshold", fallback=60)

frames = []

while True:
    frame = video_capture.read()
    if frame is None:
        continue
    frames.append(frame)
    if len(frames) >= 4:
        break

file = snapshot.generate(frames, [
    _("GENERATED SNAPSHOT"),
    _("Date: ") + datetime.now(timezone.utc).strftime("%Y/%m/%d %H:%M:%S UTC"),
    _("Dark threshold config: ") + str(config.getfloat("video", "dark_threshold", fallback=60.0)),
    _("Certainty config: ") + str(config.getfloat("video", "certainty", fallback=3.5)),
])

print(_("Generated snapshot saved as"))
print(file)
