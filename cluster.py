import math
import os
import random
import cv2
import numpy as np  # ใช้แค่สร้างภาพให้ cv2.imshow เท่านั้น ไม่ใช้คำนวณ
from face_utils import load_dataset

k = 4  # จำนวนกลุ่ม K-means (ใช้เมื่อ AUTO_K = False)
AUTO_K = False  # True = ลองทุกค่า k ใน K_RANGE แล้วเลือกค่าที่ silhouette เฉลี่ยสูงสุด
K_RANGE = range(2, 7)
SEED = 42
N_TRIES = 10  # จำนวนครั้งที่สุ่มจุดเริ่มต้นใหม่ (เลือกครั้งที่ผลรวมระยะน้อยสุด)
MAX_ITER = 100
EPS = 0.01
SAVE_IMAGE = False  # True = บันทึกภาพผลเป็น result_groups.png ไว้ใส่รายงาน (ค่าเริ่มต้นไม่บันทึกไฟล์ใด ๆ)


# ---------- ฟังก์ชันคำนวณมือทั้งหมด ----------
def mean_vectors(vectors):
    """ค่าเฉลี่ยทีละพิกเซลของหลายเวกเตอร์ (ตัดเศษทิ้งเป็นจำนวนเต็ม 0-255)"""
    n = len(vectors)
    out = []
    for i in range(len(vectors[0])):
        s = 0
        for v in vectors:
            s += v[i]
        out.append(int(s / n))
    return out


def equalize_hist(pixels):
    """ปรับ histogram ให้เท่ากัน (สูตรเดียวกับ cv2.equalizeHist)"""
    hist = [0] * 256
    for p in pixels:
        hist[p] += 1

    # ผลรวมสะสม (CDF)
    cdf = [0] * 256
    run = 0
    for i in range(256):
        run += hist[i]
        cdf[i] = run

    cdf_min = 0
    for c in cdf:
        if c > 0:
            cdf_min = c
            break

    total = len(pixels)
    if total == cdf_min:  # ภาพสีเดียว
        return list(pixels)

    lut = [0] * 256
    for i in range(256):
        val = (cdf[i] - cdf_min) * 255.0 / (total - cdf_min)
        v = int(val + 0.5)
        lut[i] = 0 if v < 0 else 255 if v > 255 else v
    return [lut[p] for p in pixels]


def sq_dist(a, b):
    """ระยะทางยูคลิดยกกำลังสอง"""
    s = 0.0
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def kmeans_pp_init(X, k, rng):
    """เลือกจุดศูนย์กลางเริ่มต้นแบบ k-means++"""
    centers = [list(X[rng.randrange(len(X))])]
    while len(centers) < k:
        d2 = []
        for x in X:
            best = sq_dist(x, centers[0])
            for c in centers[1:]:
                d = sq_dist(x, c)
                if d < best:
                    best = d
            d2.append(best)
        total = 0.0
        for d in d2:
            total += d
        r = rng.random() * total
        acc = 0.0
        pick = len(X) - 1
        for i, d in enumerate(d2):
            acc += d
            if acc >= r:
                pick = i
                break
        centers.append(list(X[pick]))
    return centers


def kmeans_once(X, k, rng):
    dim = len(X[0])
    centers = kmeans_pp_init(X, k, rng)
    labels = [-1] * len(X)

    for _ in range(MAX_ITER):
        # 1) กำหนดแต่ละจุดให้ศูนย์กลางที่ใกล้สุด
        dists = []
        for i, x in enumerate(X):
            best_g, best_d = 0, sq_dist(x, centers[0])
            for g in range(1, k):
                d = sq_dist(x, centers[g])
                if d < best_d:
                    best_g, best_d = g, d
            labels[i] = best_g
            dists.append(best_d)

        # 2) คำนวณศูนย์กลางใหม่ = ค่าเฉลี่ยของสมาชิก
        new_centers = []
        for g in range(k):
            members = [X[i] for i in range(len(X)) if labels[i] == g]
            if not members:
                # กลุ่มว่าง: ย้ายไปจุดที่อยู่ไกลศูนย์กลางเดิมมากที่สุด
                far = 0
                for i in range(len(X)):
                    if dists[i] > dists[far]:
                        far = i
                dists[far] = 0.0
                labels[far] = g
                new_centers.append(list(X[far]))
                continue
            c = [0.0] * dim
            for m in members:
                for j in range(dim):
                    c[j] += m[j]
            n = len(members)
            new_centers.append([v / n for v in c])

        # 3) หยุดเมื่อศูนย์กลางขยับน้อยกว่า EPS
        moved = 0.0
        for g in range(k):
            d = sq_dist(centers[g], new_centers[g])
            if d > moved:
                moved = d
        centers = new_centers
        if moved <= EPS * EPS:
            break

    # ผลรวมระยะทางกำลังสองของทุกจุดถึงศูนย์กลางของตัวเอง
    compact = 0.0
    for i, x in enumerate(X):
        compact += sq_dist(x, centers[labels[i]])
    return labels, compact


def kmeans(X, k, tries, seed):
    rng = random.Random(seed)
    best_labels, best_c = None, None
    for _ in range(tries):
        lab, c = kmeans_once(X, k, rng)
        if best_c is None or c < best_c:
            best_labels, best_c = list(lab), c
    return best_labels


def pairwise_dist(X):
    """ตารางระยะยูคลิดระหว่างทุกคู่ (คำนวณครั้งเดียว ใช้ซ้ำได้)"""
    n = len(X)
    D = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = math.sqrt(sq_dist(X[i], X[j]))
    return D


def silhouette(D, groups, k):
    """ค่า silhouette ของแต่ละคน: (b - a) / max(a, b)
    a = ระยะเฉลี่ยถึงเพื่อนในกลุ่มเดียวกัน, b = ระยะเฉลี่ยถึงกลุ่มอื่นที่ใกล้ที่สุด
    ใกล้ 1 = อยู่ถูกกลุ่มชัดเจน, ใกล้ 0 = อยู่กึ่งกลางระหว่างกลุ่ม, ติดลบ = ใกล้กลุ่มอื่นมากกว่า"""
    n = len(D)
    s = [0.0] * n
    for i in range(n):
        same = [D[i][j] for j in range(n) if j != i and groups[j] == groups[i]]
        if not same:  # กลุ่มที่มีคนเดียว กำหนดเป็น 0
            continue
        a = sum(same) / len(same)
        b = None
        for g in range(k):
            other = [D[i][j] for j in range(n) if groups[j] == g]
            if g != groups[i] and other:
                m = sum(other) / len(other)
                if b is None or m < b:
                    b = m
        if b is not None and max(a, b) > 0:
            s[i] = (b - a) / max(a, b)
    return s


def display_names(people):
    """ชื่อที่แสดง = ส่วนหลัง "_" ถ้าชื่อซ้ำกันให้ต่อท้ายด้วยเลข 4 ตัวท้ายของรหัส เช่น First(0607)"""
    short = [p.split("_", 1)[-1] for p in people]
    out = []
    for p, s in zip(people, short):
        if short.count(s) > 1:
            s = f"{s}({p.split('_', 1)[0][-4:]})"
        out.append(s)
    return out


# ---------- โหลดข้อมูล ----------
faces, labels = load_dataset()
people = sorted(set(labels))

# หน้าเฉลี่ยต่อ 1 คน (เก็บเป็นลิสต์ 2 มิติ + เวกเตอร์แบนสำหรับคำนวณ)
H, W = faces[0].shape
flat_faces = []
for p in people:
    vecs = [faces[i].flatten().tolist() for i in range(len(faces)) if labels[i] == p]
    flat_faces.append(mean_vectors(vecs))
names = display_names(people)

# ---------- จัดกลุ่มด้วย K-means ----------
X = [equalize_hist(f) for f in flat_faces]
D = pairwise_dist(X)  # ใช้คำนวณ silhouette

if AUTO_K:
    # ลองหลายค่า k แล้วเลือกค่าที่ silhouette เฉลี่ยสูงสุด
    print("เลือกจำนวนกลุ่มอัตโนมัติ:")
    best = None
    for kk in K_RANGE:
        if kk > len(X):
            break
        g_try = kmeans(X, kk, N_TRIES, SEED)
        s_try = silhouette(D, g_try, kk)
        score = sum(s_try) / len(s_try)
        print(f"  k = {kk}: silhouette เฉลี่ย {score:.3f}")
        if best is None or score > best[0]:
            best = (score, kk, g_try, s_try)
    _, k, groups, sil = best
    print(f"  -> ใช้ k = {k}\n")
else:
    groups = kmeans(X, k, N_TRIES, SEED)
    sil = silhouette(D, groups, k)

# แสดงรายชื่อสมาชิกแต่ละกลุ่มใน Console
for g in range(k):
    m = [i for i in range(len(people)) if groups[i] == g]
    avg = sum(sil[i] for i in m) / len(m) if m else 0.0
    print(f"กลุ่ม {g + 1} ({len(m)} คน, silhouette {avg:.3f}): " + ", ".join(names[i] for i in m))
print(f"\nsilhouette เฉลี่ยทั้งหมด: {sum(sil) / len(sil):.3f}  (ใกล้ 1 = แยกกลุ่มชัด, ใกล้ 0 = ไม่ชัด)")
odd = [f"{names[i]} ({sil[i]:.3f})" for i in range(len(people)) if sil[i] < 0]
if odd:
    print("คนที่ใกล้กลุ่มอื่นมากกว่ากลุ่มตัวเอง: " + ", ".join(odd))


# ---------- สร้างภาพสรุปผล (ใช้ cv2 วาดเท่านั้น) ----------
def to_img(flat):
    return np.array(flat, np.uint8).reshape(H, W)


def thumb(flat, txt, sub=""):
    card = np.full((146, 90), 255, np.uint8)
    card[3:111, 3:87] = cv2.resize(to_img(flat), (84, 108))
    cv2.putText(card, txt[:12], (3, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    if sub:
        cv2.putText(card, sub, (3, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.35, 90, 1)
    return card


rows = []
for g in range(k):
    m = [i for i in range(len(groups)) if groups[i] == g]
    if not m:
        continue

    # หัวแถว = หน้าเฉลี่ยของกลุ่ม (ช่วยดูว่ากลุ่มนี้มีลักษณะร่วมอะไร) + silhouette เฉลี่ยของกลุ่ม
    avg = sum(sil[i] for i in m) / len(m)
    head = thumb(mean_vectors([flat_faces[i] for i in m]), f"Group {g + 1}", f"avg s={avg:.2f}")
    cv2.rectangle(head, (1, 1), (88, 112), 0, 2)
    gap = np.full((146, 10), 255, np.uint8)

    rows.append(np.hstack([head, gap] + [thumb(flat_faces[i], names[i], f"s={sil[i]:.2f}") for i in m]))

width = max(r.shape[1] for r in rows)
result = np.vstack([np.pad(r, ((0, 6), (0, width - r.shape[1])), constant_values=255) for r in rows])

if SAVE_IMAGE:
    # ใช้ imencode + tofile แทน cv2.imwrite เพื่อให้บันทึกได้แม้ path มีภาษาไทย (Windows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_groups.png")
    cv2.imencode(".png", result)[1].tofile(out)
    print(f"\nบันทึกภาพผลที่ {out}")

cv2.imshow("K-means face groups", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
