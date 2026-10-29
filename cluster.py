import cv2
import numpy as np
from face_utils import load_dataset

k = 4  # จำนวนกลุ่ม K-means

# โหลดข้อมูลและหาภาพใบหน้าเฉลี่ยต่อ 1 คน
faces, labels = load_dataset()
labels = np.array(labels)
people = sorted(set(labels))
faces = [np.array(faces)[labels == p].mean(axis=0).astype(np.uint8) for p in people]
names = [p.split("_", 1)[-1] for p in people]

# จัดกลุ่มด้วย K-means
X = np.array([cv2.equalizeHist(f).ravel() for f in faces], np.float32)
cv2.setRNGSeed(42)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.01)
_, groups, _ = cv2.kmeans(X, k, None, criteria, 30, cv2.KMEANS_PP_CENTERS)
groups = groups.ravel()

# แสดงรายชื่อสมาชิกแต่ละกลุ่มใน Console
for g in range(k):
    members = [names[i] for i in range(len(people)) if groups[i] == g]
    print(f"กลุ่ม {g + 1} ({len(members)} คน): " + ", ".join(members))


# สร้างภาพสรุปผลแต่ละกลุ่ม
def thumb(img, txt):
    card = np.full((132, 90), 255, np.uint8)
    card[3:111, 3:87] = cv2.resize(img, (84, 108))
    cv2.putText(card, txt[:12], (3, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 0, 1)
    return card


rows = []
for g in range(k):
    m = [i for i, gg in enumerate(groups) if gg == g]
    if not m:
        continue

    head = np.full((132, 100), 255, np.uint8)
    cv2.putText(head, f"Group {g + 1}", (6, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)

    rows.append(np.hstack([head] + [thumb(faces[i], names[i]) for i in m]))

width = max(r.shape[1] for r in rows)
result = np.vstack([np.pad(r, ((0, 6), (0, width - r.shape[1])), constant_values=255) for r in rows])

cv2.imshow("K-means face groups", result)
cv2.waitKey(0)
cv2.destroyAllWindows()