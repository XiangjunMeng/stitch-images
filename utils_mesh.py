import numpy as np
import cv2

def load_mesh_data(tri_path, coord_path):
    triangles = np.load(tri_path)
    coords = np.load(coord_path)
    return triangles, coords

# def apply_mesh_transform(img, triangles, original_vertices, new_vertices, canvas_size):
def apply_mesh_transform(img, triangles, original_vertices, new_vertices):
    # h_canvas, w_canvas = canvas_size
    # warped = np.zeros((h_canvas, w_canvas, 3), dtype=np.uint8) 
    warped = np.zeros((2000, 3000, 3), dtype=np.uint8) 
    h, w = img.shape[:2]
    v_dict = {tuple(v): i for i, v in enumerate(original_vertices)}
    for tri in triangles:
        try:
            V0, V1, V2 = tri
            idx0 = v_dict[tuple(V0)]
            idx1 = v_dict[tuple(V1)]
            idx2 = v_dict[tuple(V2)]

            src = np.float32([V0, V1, V2])
            dst = np.float32([new_vertices[idx0], new_vertices[idx1], new_vertices[idx2]])

            M = cv2.getAffineTransform(src, dst)
            mask = np.zeros((h, w), dtype = np.uint8)
            cv2.fillConvexPoly(mask, np.int32(src), 255)
            warped_part = cv2.warpAffine(img, M, (3000, 2000))
            warped[mask > 0] = warped_part[mask > 0]
        except Exception as e:
            continue
    
    return warped

def build_affine_energy(triangles, coords, verts, offset):
    v_dict = {tuple(v.tolist()): i for i, v in enumerate(verts)}
    rows, cols, data, b = [], [], [], []
    row_idx = 0

    for tri, (x01, y01) in zip(triangles, coords):
        V0, V1, V2 = tri
        i0 = v_dict[tuple(V0.tolist())]
        i1 = v_dict[tuple(V1.tolist())]
        i2 = v_dict[tuple(V2.tolist())]

        vec = V1 - V0
        rot90 = np.array([-vec[1], vec[0]])
        desired = V0 + x01 * vec + y01 * rot90

        indices = [i2, i0, i1]
        targets = [desired, V0, V1]

        for (vi, target) in zip(indices, targets):
            for d in range(2):
                rows.append(row_idx)
                cols.append((offset + vi) * 2 + d)
                data.append(1.0)
                b.append(target[d])
                row_idx += 1

    return rows, cols, data, b, row_idx

def build_similarity_energy(triangles, verts, offset):
    v_dict = {tuple(v.tolist()) for i, v in enumerate(verts)}
    v_dict = {tuple(v.tolist()): i for i, v in enumerate(verts)}

    rows, cols, data, b = [], [], [], []
    row_idx = 0

    for tri in triangles:
        v0, v1, v2 = tri
        i0 = v_dict[tuple(v0.tolist())]
        i1 = v_dict[tuple(v1.tolist())]
        i2 = v_dict[tuple(v2.tolist())]

        vec01 = v1 - v0
        vec02 = v2 - v1

        x1, y1 = vec01
        x2, y2 = -y1, x1

        for dx, dy, j in [(x1, y1, i1), (x2, y2, i2)]:
            for d in range(2):
                rows.append(row_idx)
                cols.append((offset + j) * 2 + d)
                data.append(1.0)
                rows.append(row_idx)
                cols.append((offset + i0) * 2 + d)
                data.append(-1.0)
                b.append(dx if d == 0 else dy)
                row_idx += 1

    return rows, cols, data, b, row_idx
                             