import os
import sys


class VideoCapture:
    """A clean, streamlined wrapper around OpenCV's VideoCapture engine."""

    def __init__(self, config):
        # Move OpenCV import here to achieve Phase 2 Deferred Imports!
        global cv2
        import cv2

        self.config = config

        # Determine the video device path from config
        device_path = config.get("video", "device_path")
        if device_path.isdigit():
            device_path = int(device_path)

        # Enforce modern V4L2 backend selection natively
        self.internal = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

        if not self.internal.isOpened():
            print(f"Error: Could not open video device {device_path}", file=sys.stderr)
            sys.exit(1)

    def read(self):
        """Grabs, decodes and returns the next video frame."""
        ret, frame = self.internal.read()
        return frame if ret else None

    def release(self):
        """Closes the video file stream or capturing device."""
        if self.internal:
            self.internal.release()
