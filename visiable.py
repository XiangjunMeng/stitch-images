import os
import cv2
import numpy as np


image_dir ="c:/Users/robot/Desktop/bridgeModels/00002/bridgeModel/pictureset_0/in"
mesh_dir = "./triangle_meshes"
save_dir = "./images_with_triangles"
os.makedirs(save_dir, exist_ok=True)

def draw_triangles_on_image(img, triangles, color=(0, 255, 0), thickness=1):
    vis = img.copy()
    for tri in triangles:
        pts = np.int32(tri)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
    return vis


image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])
for img_name in image_files:
    name = os.path.splitext(img_name)[0]
    img_path = os.path.join(image_dir, img_name)
    tri_path = os.path.join(mesh_dir, f"{name}_triangles.npy")

    if not os.path.exists(tri_path):
        print(f"skip:{img_name}")
        continue

    img = cv2.imread(img_path)
    triangles = np.load(tri_path)  # shape: (N, 3, 2)

    vis_img = draw_triangles_on_image(img, triangles)
    save_path = os.path.join(save_dir, f"{name}_triangles.jpg")
    cv2.imwrite(save_path, vis_img)
    print(f"finished!: {save_path}")
