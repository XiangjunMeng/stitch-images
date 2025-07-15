import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr

from utils_mesh import load_mesh_data, apply_mesh_transform, build_affine_energy, build_similarity_energy

image_dir ="c:/Users/robot/Desktop/bridgeModels/00002/bridgeModel/pictureset_0/in"
mesh_dir = "./triangle_meshes"
output_dir = "./stitched_results"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".jpg")])

all_imgs, all_triangles, all_coords, all_verts = [], [], [], []
for fname in image_files:
    name = os.path.splitext(fname)[0]
    img_path = os.path.join(image_dir, fname)
    tri_path = os.path.join(mesh_dir, f"{name}_triangles.npy")
    coord_path = os.path.join(mesh_dir, f"{name}_coords.npy")

    if not os.path.exists(tri_path) or not os.path.exists(coord_path):
        continue

    img = cv2.imread(img_path)
    triangles, coords = load_mesh_data(tri_path, coord_path)
    verts = np.unique(np.vstack(triangles), axis=0)

    all_imgs.append(img)
    all_triangles.append(triangles)
    all_coords.append(coords)
    all_verts.append(verts)

print(f"total {len(all_imgs)} images")

vertex_offset = []
total_vertices = 0
for verts in all_verts:
    vertex_offset.append(total_vertices)
    total_vertices += len(verts)

rows, cols, data, b = [], [], [], []
row_counter = 0

λl = 0.75     
λobj = 1.5    

for i in range(len(all_imgs)):
    triangles = all_triangles[i]
    coords = all_coords[i]
    verts = all_verts[i]
    offset = vertex_offset[i]

    r1, c1, d1, b1, rc1 = build_affine_energy(triangles, coords, verts, offset)
    r2, c2, d2, b2, rc2 = build_similarity_energy(triangles, verts, offset)

    rows += r1 + [r + row_counter + rc1 for r in r2]
    cols += c1 + c2
    data += d1 + [λl * val for val in d2]
    b += b1 + [λl * val for val in b2]

    row_counter += rc1 + rc2

A = lil_matrix((row_counter, total_vertices * 2))
A[rows, cols] = data
b = np.array(b)

print(f" A.shape = {A.shape}, b.shape = {b.shape}")

x = lsqr(A.tocsr(), b)[0] 

canvas = np.zeros((2000, 3000, 3), dtype=np.uint8)

for i in range(len(all_imgs)):
    img = all_imgs[i]
    triangles = all_triangles[i]
    verts = all_verts[i]
    offset = vertex_offset[i]

    new_verts = []
    for j in range(len(verts)):
        new_verts.append([
            x[(offset + j) * 2],
            x[(offset + j) * 2 + 1]
        ])

    warped = apply_mesh_transform(img, triangles, verts, new_verts)
    canvas = np.maximum(canvas, warped)

cv2.imwrite(os.path.join(output_dir, "stitched_niswgsp_like.jpg"), canvas)
print("finished！")
