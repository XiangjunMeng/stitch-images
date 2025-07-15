import numpy as np
import matplotlib.pyplot as plt
import os

contour_dir = "./contours"
mesh_output_dir = "./triangle_meshes"
vis_output_dir = "./mesh_vis"

os.makedirs(mesh_output_dir, exist_ok=True)
os.makedirs(vis_output_dir, exist_ok=True)

def build_mesh_from_contour(contour, sampling_step=10):
    V0 = np.mean(contour, axis = 0)
    sampled = contour[::sampling_step]
    triangles = []
    local_coords = []

    for i in range(len(sampled) - 1):
        V1 = sampled[i]
        V2 = sampled[i + 1]

        vec = V1 - V0
        R90 = np.array([-vec[1], vec[0]])
        A = np.stack([vec, R90], axis = 1)
        b = V2 - V0

        try:
            xy = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            continue

        triangles.append([V0, V1, V2])
        local_coords.append(xy)
    
    return np.array(triangles), np.array(local_coords)

for fname in os.listdir(contour_dir):
    if not fname.endswith("_contour.npy"):
        continue

    name = fname.replace("_contour.npy", "")
    contour_path = os.path.join(contour_dir, fname)
    contour = np.load(contour_path)

    if len(contour) < 20:
        print("skipped")
        continue

    triangles, coords = build_mesh_from_contour(contour)
    
    np.save(os.path.join(mesh_output_dir, f"{name}_triangles.npy"), triangles)
    np.save(os.path.join(mesh_output_dir, f"{name}_coords.npy"), coords)

    plt.figure(figsize=(6, 6))
    for tri in triangles:
        pts = np.vstack((tri, tri[0]))
        plt.plot(pts[:, 0], pts[:, 1], 'b-')
    plt.plot(*np.mean(contour, axis = 0), 'ro', label = 'Center')
    plt.title(f"triangle mesh: {name}")
    plt.axis('equal')
    plt.savefig(os.path.join(vis_output_dir, f"{name}_mesh.png"))
    plt.close()
