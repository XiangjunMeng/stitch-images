import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

image_dir = 'c:/Users/robot/Desktop/bridgeModels/00004/pictureset_0/in/mvmpr/images'
projection_dir = 'c:/Users/robot/Desktop/bridgeModels/00004/pictureset_0/in/mvmpr/data'
plane = np.array([-0.0370, 0.0507, -0.09980, 18.9493])
canvas_size = (4000, 4000)
canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype = np.uint8)
print(canvas.shape[0])
scale = 100
offset = np.array([2000, 2000])

def load_projection_matrix(txt_path):
    with open(txt_path, 'r', encoding = 'latin1') as f:
        lines = f.readlines()

    numeric_lines = []
    for line in lines:
        parts = ''.join(c for c in line if c.isprintable()).strip().split()
        try:
            row = [float(p) for p in parts]
            if len(row) == 4:
                numeric_lines.append(row)
        except ValueError:
            continue

    if len(numeric_lines) != 3:
        raise ValueError(f"{txt_path} does not contain a valid projection matrix")
    
    return np.array(numeric_lines)


def project_pixel_to_plane(P, u, v, plane_eq):
    A, B, C, D = plane_eq
    P = np.asarray(P).squeeze()
    U, S, Vt = np.linalg.svd(P)
    C_cam = Vt[-1]
    C_cam = (C_cam / C_cam[-1])[:3]
    pixel = np.array([u, v, 1.0])
    A_mat = P[:, :3]
    # print(A_mat.shape)
    b_vec = P[:, 3]
    # print(b_vec.shape)12  
    try:
        d = np.linalg.pinv(A_mat) @ (pixel - b_vec)
    except np.linalg.LinAlgError:
        return None
    
    d = d / np.linalg.norm(d)
    denom = A * d[0] + B * d[1] + C * d[2]
    if abs(denom) < 1e-6:
        return None
    
    t = -(A * C_cam[0] + B * C_cam[1] + C * C_cam[2] + D) / denom
    X = C_cam + t * d
    return X

image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.png','.jpg'))])
projection_paths = sorted([os.path.join(projection_dir, f) for f in os.listdir(projection_dir) if f.endswith('.txt')])

assert len(image_paths) == len(projection_paths), "mismatch between images and projection files"

for img_path, proj_path in zip(image_paths, projection_paths):
    print(f"processing:{os.path.basename(img_path)}")
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    P = load_projection_matrix(proj_path)

    for v in range(0, h):
        for u in range(0, w):
            color = img[v, u]
            pt = project_pixel_to_plane(P, u, v, plane)
            if pt is not None:
                x, y = pt[0], pt[1]
                px = int(x * scale + offset[0])
                py = int(y * scale + offset[1])
                # print(px, py)
                if 0 <= px < canvas.shape[1] and 0 <= py < canvas.shape[0]:
                    canvas[py, px] = color

plt.figure(figsize=(12, 12))
plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
plt.title("projected stitching on plane")
plt.axis("off")
plt.tight_layout()
plt.show()
