import os
import re
import cv2
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
IMG_SIZE = (140, 180)  # (กว้าง, สูง) ของหน้าที่ตัดแล้ว
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_from_photo(gray):
    """รูปนักเรียน -> หน้าขาวดำช่วงหน้าผาก-คาง แก้ม-แก้ม ขนาด IMG_SIZE (None ถ้าหาหน้าไม่เจอ)"""
    # ตัดขอบสีเรียบออก
    for _ in range(4):
        if gray[0].std() < 4: gray = gray[1:]
        if gray[-1].std() < 4: gray = gray[:-1]
        if gray[:, 0].std() < 4: gray = gray[:, 1:]
        if gray[:, -1].std() < 4: gray = gray[:, :-1]

    h, w = gray.shape
    gray = cv2.resize(gray, (round(w * (600 / max(h, w))), round(h * (600 / max(h, w)))), interpolation=cv2.INTER_AREA)
    gray = cv2.copyMakeBorder(gray, 150, 150, 150, 150, cv2.BORDER_REPLICATE)

    faces = CASCADE.detectMultiScale(gray, 1.05, 6, minSize=(80, 80))
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    cw = w * 0.78
    x0, y0 = int(round(x + (w - cw) / 2)), int(round(y + h * 0.06))
    x1, y1 = x0 + int(round(cw)), y0 + int(round(cw * IMG_SIZE[1] / IMG_SIZE[0]))

    p = max(0, -x0, -y0, x1 - gray.shape[1], y1 - gray.shape[0])
    if p > 0:
        gray = cv2.copyMakeBorder(gray, p, p, p, p, cv2.BORDER_REPLICATE)

    return cv2.resize(gray[y0 + p:y1 + p, x0 + p:x1 + p], IMG_SIZE, interpolation=cv2.INTER_AREA)


def load_dataset():
    """อ่านทุกรูปใน dataset/ -> (หน้าที่ตัดแล้ว, ชื่อ)"""
    faces, labels = [], []
    for file in sorted(os.listdir(DATASET_DIR)):
        path = os.path.join(DATASET_DIR, file)
        img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)

        face = face_from_photo(img) if img is not None else None
        if face is None:
            print(f"  ข้าม (อ่านไม่ได้หรือหาหน้าไม่เจอ): {file}")
            continue

        faces.append(face)
        labels.append(re.sub(r"[\s_\-]*\(?\d+\)?$", "", os.path.splitext(file)[0]))

    return faces, labels