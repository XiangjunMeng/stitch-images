import os 
import cv2
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
from utils_mesh import (
    load_mesh_data, 
    apply_mesh_transform,
    build_affine_energy,
    build_similarity_energy
) 

image_dir = "c:/Users/robot/Desktop/bridgeModels/00002/bridgeModel/pictureset_0/in"
mesh_dir = "./triangle_meshes"
output_dir = "./stitched_results"
os.makedirs(output_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
all_imgs = []
all_triangles = []
all_coords = []
all_names = []
for fname in image_files:
    name = os.path.splitext(fname)[0]
    img_path = os.path.join(image_dir, fname)
    tri_path = os.path.join(mesh_dir, f"{name}_triangles.npy")
    coord_path = os.path.join(mesh_dir, f"{name}_coords.npy")

    if not os.path.exists(tri_path) or not os.path.exists(coord_path):
        continue
    img = cv2.imread(img_path)
    triangles, coords = load_mesh_data(tri_path, coord_path)

    all_imgs.append(img)
    all_triangles.append(triangles)
    all_coords.append(coords)
    all_names.append(name)
print(f"total is {len(all_imgs)} pictures load successfully and ready to optimate")

vertex_offset = []
all_vertices = []
total_vertices = 0
for tri in all_triangles:
    verts = np.unique(np.vstack(tri), axis = 0)
    all_vertices.append(verts)
    vertex_offset.append(total_vertices)
    total_vertices += len(verts)

rows = []
cols = []
data = []
b = []

row_counter = 0

for img_id, (triangles, coords, verts) in enumerate(zip(all_triangles, all_coords, all_vertices)):
    offset = vertex_offset[img_id]
    r1, c1, d1, b1, rc1 = build_affine_energy(triangles, coords, verts, offset)
    r2, c2, d2, b2, rc2 = build_similarity_energy(triangles, verts, offset)

    rows += r1 + [r +row_counter + rc1 for r in r2]
    cols += c1 + c2
    data += d1 + d2
    b += b1 + b2
    row_counter += rc1 + rc2

A = lil_matrix((row_counter, total_vertices * 2))
A[rows, cols] = data
b = np.array(b)
print(f"A.shpae = {A.shape}, b.shape = {b.shape}")

x = lsqr(A.tocsr(), b)[0]

x_coords = x[::2]
y_coords = x[1::2]
min_x, max_x = np.min(x_coords), np.max(x_coords)
min_y, max_y = np.min(y_coords), np.max(y_coords)

target_width = 3000
target_height = 2000

scale_x = target_width / (max_x - min_x)
scale_y = target_height / (max_y - min_y)
scale = min(scale_x, scale_y) * 0.95

x_coords = (x_coords - min_x) * scale + 50
y_coords = (y_coords - min_y) * scale + 50

for i in range(len(x_coords)):
    x[2 * i] = x_coords[i]
    x[2 * i + 1] = y_coords[i]

max_h, max_w = 0, 0
for img in all_imgs:
    h, w = img.shape[:2]
    max_h = max(max_h, h)
    max_w = max(max_w, w)

canvas_size = (max_h + 200, max_w + 200)
canvas = np.zeros((canvas_size[0], canvas_size[1], 3), dtype = np.uint8)

for img_id, (img, triangles, verts) in enumerate(zip(all_imgs, all_triangles, all_vertices)):
    offset = vertex_offset[img_id]
    v_dict = {tuple(v): i for i, v in enumerate(verts)}

    new_verts = []
    for i in range(len(verts)):
        x0 = x[(offset + i) * 2]
        y0 = x[(offset + i) * 2]
        new_verts.append([x0, y0])

    warped = apply_mesh_transform(img, triangles, verts, new_verts, canvas_size=canvas_size)
    canvas = np.maximum(canvas, warped)

out_path = os.path.join(output_dir, "stitched_objgsp.jpg")
cv2.imwrite(out_path, canvas)
print(f"finished")
