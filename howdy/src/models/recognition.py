import os
import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

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


def match_faces(
    saved_embedding: np.ndarray, current_embedding: np.ndarray, threshold: float = 0.75
) -> bool:
    dot_product = np.dot(saved_embedding, current_embedding)
    norm_saved = np.linalg.norm(saved_embedding)
    norm_current = np.linalg.norm(current_embedding)
    similarity = dot_product / (norm_saved * norm_current)
    return float(similarity) > threshold
