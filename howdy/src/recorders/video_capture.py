import os
import sys


class VideoCapture:
    def __init__(self, config):
        global cv2
        import cv2

        raw_paths = config.get("video", "device_path")
        paths = [p.strip() for p in raw_paths.split(",")]

        self.internal = None
        for device_path in paths:
            if device_path.isdigit():
                device_path = int(device_path)
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
            if cap.isOpened():
                self.internal = cap
                break
            cap.release()

        if self.internal is None:
            print(f"Error: Could not open any video device from '{raw_paths}'",
                  file=sys.stderr)
            sys.exit(1)

        force_mjpeg = config.getboolean("video", "force_mjpeg", fallback=False)
        if force_mjpeg:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.internal.set(cv2.CAP_PROP_FOURCC, fourcc)

        device_fps = config.getint("video", "device_fps", fallback=-1)
        if device_fps > 0:
            self.internal.set(cv2.CAP_PROP_FPS, device_fps)

        fw = config.getint("video", "frame_width", fallback=-1)
        fh = config.getint("video", "frame_height", fallback=-1)
        if fw > 0 and fh > 0:
            self.internal.set(cv2.CAP_PROP_FRAME_WIDTH, fw)
            self.internal.set(cv2.CAP_PROP_FRAME_HEIGHT, fh)

    def read(self):
        """Grabs, decodes and returns the next video frame."""
        ret, frame = self.internal.read()
        return frame if ret else None

    def set(self, prop: int, value: float):
        """Sets a property on the underlying VideoCapture."""
        self.internal.set(prop, value)

    def release(self):
        """Closes the video file stream or capturing device."""
        if self.internal:
            self.internal.release()
