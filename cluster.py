import os
import numpy as np
from PIL import Image, ImageDraw
from face_utils import load_dataset

k = 3  # จำนวนกลุ่ม

# 1. โหลดข้อมูล แล้วเฉลี่ยภาพของแต่ละคนให้เหลือคนละ 1 ภาพ
faces, labels = load_dataset()
faces, labels = np.array(faces), np.array(labels)
people = sorted(set(labels))
raw_faces = [faces[labels == p].mean(axis=0).astype(np.uint8) for p in people]
names = [p.split("_", 1)[-1] for p in people]

# 2. Histogram Equalization (เขียนเอง)
def equalize(img):
    cdf = np.bincount(img.ravel(), minlength=256).cumsum()
    lo = cdf[cdf > 0].min()
    return (cdf[img] - lo) * 255 // (cdf[-1] - lo)

X = np.array([equalize(f).ravel() for f in raw_faces], np.float32)

# 3. K-means (เขียนเอง)
def kmeans(X, k, iters=100, seed=42):
    np.random.seed(seed)
    c = X[np.random.choice(len(X), k, replace=False)]
    for _ in range(iters):
        g = np.linalg.norm(X[:, None] - c, axis=2).argmin(axis=1)
        new_c = np.array([X[g == i].mean(axis=0) if (g == i).any() else c[i] for i in range(k)])
        if np.allclose(c, new_c):
            break
        c = new_c
    return g

groups = kmeans(X, k)
members = [[i for i in range(len(names)) if groups[i] == g] for g in range(k)]

# 4. แสดงรายชื่อใน console และวาดภาพสรุปลงผืนเดียว
canvas = Image.new("L", (100 + 90 * max(map(len, members)), 138 * k), 255)
draw = ImageDraw.Draw(canvas)
for g, m in enumerate(members):
    print(f"กลุ่ม {g + 1} ({len(m)} คน): " + ", ".join(names[i] for i in m))
    y = g * 138
    draw.text((10, y + 60), f"Group {g + 1}", fill=0)
    for j, i in enumerate(m):
        x = 100 + j * 90
        canvas.paste(Image.fromarray(raw_faces[i]).resize((84, 108)), (x + 3, y + 3))
        draw.text((x + 3, y + 114), names[i][:12], fill=0)

canvas.save("kmeans_result.jpg")
os.startfile("kmeans_result.jpg")  # เปิดไฟล์ที่เซฟไว้ (Windows)