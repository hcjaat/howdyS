#!/usr/bin/env python3
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "howdy/src"))
import cv2, numpy as np
from compare import encode_face
import paths_factory

user = os.getlogin()
enc_file = paths_factory.user_model_path(user)

encodings = []
if os.path.exists(enc_file):
    encodings = json.load(open(enc_file))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print(f"Look at the camera. Capturing in 2 seconds...")
time.sleep(2)

for _ in range(15):
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: No frame"); sys.exit(1)

encoding = encode_face(frame)
if encoding is None:
    print("FAIL: No face detected"); sys.exit(1)

model = {
    "time": int(time.time()),
    "label": "test_model",
    "id": len(encodings),
    "data": [encoding.tolist()],
}
encodings.append(model)

os.makedirs(os.path.dirname(enc_file), exist_ok=True)
with open(enc_file, "w") as f:
    json.dump(encodings, f)

print(f"Enrolled! Saved to {enc_file}")
print(f"Embedding: 128 dims, norm={np.linalg.norm(encoding):.4f}")
