import os
import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

DETECTION_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet.onnx")
RECOGNITION_MODEL = os.path.join(MODELS_DIR, "face_recognition_mobilefacenet.onnx")


def encode_face(frame: np.ndarray) -> np.ndarray | None:
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]

    face_detector = cv2.FaceDetectorYN.create(DETECTION_MODEL, "", (w, h))
    _, faces = face_detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None

    x, y, box_w, box_h = map(int, faces[0][:4])
    x = max(0, x)
    y = max(0, y)

    if box_w <= 0 or box_h <= 0:
        return None

    face_crop = frame[y : y + box_h, x : x + box_w]
    face_blob = cv2.dnn.blobFromImage(
        face_crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )

    session = ort.InferenceSession(
        RECOGNITION_MODEL, providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    embedding = session.run(None, {input_name: face_blob})[0]

    return embedding.flatten()


def match_faces(
    saved_embedding: np.ndarray, current_embedding: np.ndarray, threshold: float = 0.75
) -> bool:
    dot_product = np.dot(saved_embedding, current_embedding)
    norm_saved = np.linalg.norm(saved_embedding)
    norm_current = np.linalg.norm(current_embedding)
    similarity = dot_product / (norm_saved * norm_current)
    return float(similarity) > threshold
