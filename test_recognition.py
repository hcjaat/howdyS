#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "howdy/src"))

import cv2
import numpy as np
from compare import encode_face

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera"); sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Warming up camera...")
for _ in range(15):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: No frame captured"); sys.exit(1)

print(f"Frame: {frame.shape[1]}x{frame.shape[0]}")

encoding = encode_face(frame)
if encoding is None:
    print("FAIL: No face detected. Check lighting/camera angle.")
    sys.exit(1)

print(f"OK: {len(encoding)}-dim embedding generated")
print(f"    First 8 values: {np.round(encoding[:8], 4)}")
print(f"    Norm: {np.linalg.norm(encoding):.4f}")
print()
print("Face recognition pipeline works correctly.")
