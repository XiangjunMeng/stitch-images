import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def warp_image_by_triangles(img, triangles, coords, H):
    h, w = img.shape[:2]
    warped = np.zeros_like(img)

    for tri, (x, y) in zip(triangles, coords):
        V0, V1, _ = tri
        cV0 = cv2.perspectiveTransform(V0.reshape(1, 1, 2), H)[0, 0]
        cV1 = cv2.perspectiveTransform(V1.reshape(1, 1, 2), H)[0, 0]

        vec = cV1 - cV0
        rot90 = np.array([-vec[1], vec[0]])
        cV2 = cV0 + x * vec + y * rot90
        src_tri = np.float32([V0, V1, tri[2]])
        dst_tri = np.float32([cV0, cV1, cV2])
        M = cv2.getAffineTransform(src_tri, dst_tri)
        mask = np.zeros((h, w), dtype = np.uint8)
        cv2.fillConvexPoly(mask, np.int32(src_tri), 255)
        warped_part = cv2.warpAffine(img, M, (w, h))
        warped[mask > 0] = warped_part[mask > 0]
    return warped

image_dir ="c:/Users/robot/Desktop/bridgeModels/00002/bridgeModel/pictureset_0/in"
contour_dir = "./contour"
mesh_dir = "./triangle_meshes"

img_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
stitched = None
base_H = np.eye(3)

for i in range(len(img_files)):
    img_name = img_files[i]
    img_path = os.path.join(image_dir, img_name)
    img = cv2.imread(img_path)

    print(f"processing {img_name}")

    if i == 0:
        stitched = img.copy()
        continue

    prev_img = cv2.imread(os.path.join(image_dir, img_files[i-1]))
    gray1 = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    if des1 is None or des2 is None:
        print(F"skip")
        continue
    
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]

    if len(good) < 4:
        print(f"failed")
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    if H is None:
        print(f"homography calculation failed for {img_name}")
        continue
    if base_H.shape != (3, 3) or H.shape != (3, 3):
        print(f"incorrect homography matrix shapes: base_H:{base_H.shape}, H:{H.shape}")
        continue

    name = os.path.splitext(img_name)[0]
    tri_path = os.path.join(mesh_dir, f"{name}_triangles.npy")
    coord_path = os.path.join(mesh_dir, f"{name}_coords.npy")

    if not os.path.exists(tri_path) or not os.path.exists(coord_path):
        print(f"missing")
        continue

    triangles = np.load(tri_path)
    coords = np.load(coord_path)

    base_H = base_H @ H

    warped = warp_image_by_triangles(img, triangles, coords, base_H)

    stitched = np.maximum(stitched, warped)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
plt.title("batch image stitching with mesh deformation")
plt.axis("off")
plt.tight_layout()
plt.show()
