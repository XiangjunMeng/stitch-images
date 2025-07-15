import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud("c:/Users/robot/Desktop/bridgeModels/00002/bridgeModel/pictureset_0/matching_0/triangulation_0/densification_0/PMVS/centers-all.ply")
points = np.asarray(pcd.points)

centroid = np.mean(points, axis = 0)

_, _, vh = np.linalg.svd(points - centroid)
normal = vh[2, :]

A, B, C = normal
D = -np.dot(normal, centroid)

print(f"plane equation: {A:.4f}x + {B:.4f}y + {C:.4f}z + {D:.4f} = 0")

A, B, C, D = -0.0370, 0.0507, -0.9980, 18.9493

xx, yy, = np.meshgrid(np.linspace(0, 20, 10), np.linspace(0, 20, 10))
zz = (-A * xx - B * yy - D) / C
points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

o3d.io.write_point_cloud("c:/Users/robot/Desktop/bridgeModels/fitted_plane.ply", pcd)