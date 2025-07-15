import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

image_paths = ["c:/Users/robot/Desktop/summer3d/images_dataset/0.png", "c:/Users/robot/Desktop/summer3d/images_dataset/1.png"]
projection_paths = ["c:/Users/robot/Desktop/bridgeModels/0003/pictureset_0/in/mvmpr/data/00000000.txt", "c:/Users/robot/Desktop/bridgeModels/0003/pictureset_0/in/mvmpr/data/00000001.txt"]
plane = np.array([-0.0370, 0.0507, -0.9980, 18.9493])

canvas_size = (2000, 2000)
canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
scale = 100
offset = np.array([1000, 1000])

def project_pixel_to_plane(P, u, v, plane_eq):
    A, B, C, D = plane_eq
    P = np.asarray(P).squeeze()
    U, S, Vt = np.linalg.svd(P)
    C_cam = Vt[-1]
    C_cam = (C_cam / C_cam[-1])[:3]
    pixel = np.array([u, v, 1.0])
    A_mat = P[:, :3]
    print(A_mat.shape)
    b_vec = P[:, 3]
    print(b_vec.shape)
    d = np.linalg.inv(A_mat) @ (pixel - b_vec)
    d = d / np.linalg.norm(d)
    denom = A * d[0] + B * d[1] + C * d[2]
    if abs(denom) < 1e-6:
        return None
    
    t = -(A * C_cam[0] + B * C_cam[1] + C * C_cam[2] + D) / denom
    X = C_cam + t * d
    return X

def load_projection_matrix(txt_path):
    with open(txt_path, 'r', encoding='latin1') as f:
        lines = f.readlines()

    numeric_lines = []
    for line in lines:
        clean_line = ''.join(c for c in line if c.isprintable())
        parts = clean_line.strip().split()
        try:
            row = [float(p) for p in parts]
            numeric_lines.extend(row)
        except ValueError:
            continue
    
    if len(numeric_lines) != 12:
        raise ValueError(f"file {txt_path} does not contain a valid projection matrix")
    matrix = np.array(numeric_lines).reshape(3, 4)
    return matrix

projections = []
images = []
for img_path, P_path in zip(image_paths, projection_paths):
    img = cv2.imread(img_path)
    images.append(img)
    print(np.asarray(images).shape)
    h, w  = img.shape[:2]
    print(h, w)
    P = load_projection_matrix(P_path)
    print(P)
   
    for v in range(0, h, 5):
        for u in range(0, w, 5):
            if 0 <= v < h and 0 <= u < w:
                color = img[v, u]
                pt = project_pixel_to_plane(P, u, v, plane)
                if pt is not None:
                    x, y = pt[0], pt[1]
                    px = int(x * scale + offset[0])
                    py = int(y * scale + offset[1])
                    print(px, py)
                    if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                        canvas[py, px] = color

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title("projected stitching")
plt.axis("off")
plt.tight_layout()
plt.show()