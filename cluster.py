import cv2
import numpy as np
from face_utils import load_dataset

k = 4  # กำหนดจำนวนกลุ่มโดยตรง (แทนการวนลูปหาด้วย silhouette)

# ลักษณะที่ใช้อธิบายกลุ่ม: (ถ้าสูงกว่าค่าเฉลี่ยห้อง, ถ้าต่ำกว่า)
TRAITS = [("ผิวหน้าสว่าง", "ผิวหน้าเข้ม"),
          ("หน้าผากโล่ง ไม่มีผมบัง", "มีผม/หน้าม้าปิดหน้าผาก"),
          ("ข้างแก้มโล่ง (ผมสั้น)", "มีผมคลุมข้างแก้ม (ผมยาว)"),
          ("ใส่แว่นเยอะ", "ไม่ค่อยมีคนใส่แว่น"),
          ("ภาพคมชัด เงาเข้ม", "ภาพแสงนุ่ม/ซีด")]


def part(img, y0, y1, x0, x1):
    """ตัดส่วนของหน้าตามสัดส่วน 0-1"""
    h, w = img.shape
    return img[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def measure_traits(face):
    """วัดลักษณะตาม TRAITS"""
    f = face.astype(np.float32)
    cheek = part(f, .45, .7, .25, .75).mean()
    sides = np.r_[part(f, .05, .6, 0, .08).ravel(), part(f, .05, .6, .92, 1).ravel()].mean()
    edges = np.abs(cv2.Sobel(cv2.equalizeHist(face), cv2.CV_32F, 0, 1))
    return [cheek,                                                   # ความสว่างแก้ม + จมูก
            part(f, 0, .12, .25, .75).mean() / cheek,                # หน้าผากมืด = มีผมบัง
            sides / cheek,                                           # ข้างแก้มมืด = มีผมคลุม
            part(edges, .22, .34, .44, .56).mean() / edges.mean(),  # เส้นสะพานแว่นระหว่างตา
            f.std()]                                                 # ความคมชัด


def kmeans(points, k):
    cv2.setRNGSeed(42)   # ให้ผลเหมือนเดิมทุกครั้ง
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
    _, groups, _ = cv2.kmeans(points, k, None, criteria, 30, cv2.KMEANS_PP_CENTERS)
    groups = groups.ravel()
    order = sorted(range(k), key=lambda g: -(groups == g).sum())   # กลุ่ม 1 = กลุ่มใหญ่สุด
    return np.array([order.index(g) for g in groups])


# ===== อ่านรูป: 1 คน = 1 หน้า (ถ้ามีหลายรูปใช้หน้าเฉลี่ย) =====
faces, labels = load_dataset()
labels = np.array(labels)
people = sorted(set(labels))
faces = [np.array(faces)[labels == p].mean(axis=0).astype(np.uint8) for p in people]
names = [p.split("_", 1)[-1] for p in people]
names = [f"{n}({p.split('_')[0][-4:]})" if names.count(n) > 1 else n for n, p in zip(names, people)]

# ===== K-means (ไม่ใช้ PCA และไม่ใช้ Silhouette) =====
X = np.array([cv2.equalizeHist(f).ravel() for f in faces], np.float32)   # ปรับแสงให้เท่ากันก่อน
groups = kmeans(X, k)

# ===== เกณฑ์ของแต่ละกลุ่ม: ค่าเฉลี่ยกลุ่มห่างจากค่าเฉลี่ยห้องกี่ SD (z) =====
T = np.array([measure_traits(f) for f in faces])
z = (T - T.mean(axis=0)) / T.std(axis=0)
for g in range(k):
    zg = z[groups == g].mean(axis=0)
    crit = [f"{TRAITS[j][0] if zg[j] > 0 else TRAITS[j][1]} [z={zg[j]:+.2f}]"
            for j in np.argsort(-abs(zg)) if abs(zg[j]) >= 0.5]
    print(f"\nกลุ่ม {g + 1} ({(groups == g).sum()} คน): "
          + ", ".join(n for n, gg in zip(names, groups) if gg == g))
    print("   เกณฑ์: " + (", ".join(crit) or "ไม่ต่างจากค่าเฉลี่ยห้องชัดเจน"))

d = np.linalg.norm(X[:, None] - X[None], axis=2) + np.eye(len(X)) * 1e9
a, b = np.unravel_index(d.argmin(), d.shape)
print(f"\nคู่ที่หน้าเหมือนกันที่สุดในห้อง: {names[a]} กับ {names[b]}")


# ===== ภาพสรุป: 1 แถว = 1 กลุ่ม (ชื่อกลุ่ม, หน้าเฉลี่ยของกลุ่ม, สมาชิก) =====
def thumb(img, text):
    c = np.full((126, 90), 255, np.uint8)
    c[:108, 3:87] = cv2.resize(img, (84, 108))
    cv2.putText(c, text[:12], (3, 121), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    return c


rows = []
for g in range(k):
    members = [i for i in range(len(people)) if groups[i] == g]
    avg = np.mean([faces[i] for i in members], axis=0).astype(np.uint8)
    head = np.full((126, 100), 255, np.uint8)   # ช่องหน้าสุด: ชื่อกลุ่ม
    cv2.putText(head, f"Group {g + 1}", (6, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    rows.append(np.hstack([head, thumb(avg, "average")] + [thumb(faces[i], names[i]) for i in members]))
width = max(r.shape[1] for r in rows)
result = np.vstack([np.pad(r, ((0, 6), (0, width - r.shape[1])), constant_values=255) for r in rows])
cv2.imshow("K-means face groups (press any key to close)", result)
cv2.waitKey(0)
cv2.destroyAllWindows()