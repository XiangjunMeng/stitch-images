import numpy as np
import cv2
from collections import deque, defaultdict
import os
from scipy.spatial import KDTree
from itertools import combinations
import matplotlib.pyplot as plt


image_dir = 'c:/Users/robot/Desktop/summer3d/images'
k = 4  
ratio_thresh = 0.75  
ransac_thresh = 4.0
num_ransac_trials = 500
top_m = 6  
alpha, beta = 8.0, 0.3  


sift = cv2.SIFT_create()
image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.JPG', '.PNG'))])

images, keypoints_list, descriptors_list = [], [], []
for path in image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    images.append(img)
    kp, des = sift.detectAndCompute(img, None)
    keypoints_list.append(kp)
    descriptors_list.append(des)

n_images = len(images)

new_images, new_kps, new_des = [], [], []
for i, (img, kp, des) in enumerate(zip(images, keypoints_list, descriptors_list)):
    if isinstance(des, np.ndarray) and des.ndim == 2 and des.shape[0] >= 2 and des.shape[1] > 1:
        new_images.append(img)
        new_kps.append(kp)
        new_des.append(des.astype(np.float64, copy=False))
    else:
        print(f"[drop] image {i}: bad descriptor ({None if des is None else des.shape}), skip")

images, keypoints_list, descriptors_list = new_images, new_kps, new_des
n_images = len(images)
if n_images < 2:
    raise RuntimeError("efficient pictures no more than 2（no sufficient features）。请检查图片或换更明显的纹理/降低阈值。")

trees = [KDTree(des) for des in descriptors_list]

match_graph = {}

for i, j in combinations(range(n_images), 2):
    des1, des2 = descriptors_list[i], descriptors_list[j]
    kp1, kp2 = keypoints_list[i], keypoints_list[j]

    kq = min(k, des2.shape[0])
    if kq < 2:
        continue

    good_matches = []
    for idx1, d1 in enumerate(des1):
        dists, indices = trees[j].query(d1, k=kq)
        dists = np.atleast_1d(dists)
        indices = np.atleast_1d(indices)
        if dists.size < 2:
            continue
        if dists[0] < ratio_thresh * dists[1]:
            m = cv2.DMatch(_queryIdx=idx1, _trainIdx=indices[0], _imgIdx=j, _distance=dists[0])
            good_matches.append(m)

    nf = len(good_matches)
    if nf < 4:
        continue

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh)
    if H is None:
        continue

    ni = int(np.sum(mask)) 
    if ni > alpha + beta * nf:
        match_graph[(i, j)] = {
            'H': H,
            'inliers': ni,
            'total_matches': nf,
            'matches': good_matches,
            'mask': mask.ravel().tolist()
        }
        print(f'[Accepted] Image {i} ↔ Image {j} | inliers = {ni}, total = {nf}')
    else:
        print(f'[Rejected] Image {i} ↔ Image {j} | inliers = {ni}, total = {nf}')


final_graph = {}
for i in range(n_images):
    related = []
    for (a, b), data in match_graph.items():
        if i == a:
            related.append((b, data))
        elif i == b:
            related.append((a, data))
    
    related.sort(key=lambda x: x[1]['inliers'], reverse=True)
    for target, data in related[:top_m]:
        key = (i, target) if i < target else (target, i)
        final_graph[key] = data


for (i, j), data in final_graph.items():
    img_matches = cv2.drawMatches(images[i], keypoints_list[i],
                                   images[j], keypoints_list[j],
                                   data['matches'], None,
                                   matchesMask=data['mask'],
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def get_H_i_to_j(i, j, final_graph):
    if (i, j) in final_graph:
        return final_graph[(i, j)]['H']
    elif (j, i) in final_graph:
        H_ji = final_graph[(j, i)]['H']
        H_ij = np.linalg.inv(H_ji)
        return H_ij
    else:
        return None

root = n_images // 2

adj = defaultdict(list)
for (a, b), data in final_graph.items():
    w = data['inliers']
    adj[a].append((b, w))
    adj[b].append((a, w))
for u in adj:
    adj[u].sort(key=lambda x: x[1], reverse=True)


T = {root: np.eye(3, dtype=np.float64)}
parent = {root: -1}
q = deque([root])

while q:
    u = q.popleft()
    for v, _ in adj[u]:
        if v in T:   
            continue
        H_u_to_v = get_H_i_to_j(u, v, final_graph)
        if H_u_to_v is None:
            continue
        try:
            H_v_to_u = np.linalg.inv(H_u_to_v)
        except np.linalg.LinAlgError:
            continue
        T[v] = T[u] @ H_v_to_u
        parent[v] = u
        q.append(v)

connected_ids = sorted(T.keys())
print(f'Connected images: {connected_ids} / {n_images}')

def warped_corners(img_shape, H):
    h, w = img_shape[:2]
    corners = np.array([[0,0],[w,0],[w,h],[0,h]], dtype=np.float32).reshape(-1,1,2)
    wc = cv2.perspectiveTransform(corners, H).reshape(-1,2)
    return wc

all_pts = []
for i in connected_ids:
    wc = warped_corners(images[i].shape, T[i])
    all_pts.append(wc)
all_pts = np.vstack(all_pts)

min_x, min_y = np.floor(all_pts.min(axis=0)).astype(int)
max_x, max_y = np.ceil(all_pts.max(axis=0)).astype(int)

tx, ty = -min_x, -min_y
Translate = np.array([[1,0,tx],[0,1,ty],[0,0,1]], dtype=np.float64)

canvas_w = int(max_x - min_x)
canvas_h = int(max_y - min_y)
print('Canvas size:', canvas_w, canvas_h)

acc = np.zeros((canvas_h, canvas_w), dtype=np.float64)        
wgt = np.zeros((canvas_h, canvas_w), dtype=np.float64)        

def soft_mask(img):
    mask = (img > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist

for i in connected_ids:
    H_canvas = Translate @ T[i]                 
    warped = cv2.warpPerspective(images[i], H_canvas, (canvas_w, canvas_h))
    mask = soft_mask(warped)

    acc += warped.astype(np.float64) * mask
    wgt += mask

wgt[wgt == 0] = 1.0
stitched = (acc / wgt).astype(np.uint8)

plt.figure(figsize=(16,10))
plt.title('Stitched panorama')
plt.imshow(stitched, cmap='gray')
plt.axis('off')
plt.show()
