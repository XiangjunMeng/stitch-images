Author：Xiangjun Meng

Overview: Image Stitching for Bridge Surface Reconstruction

A Python-based research project exploring multi-image stitching and mosaic reconstruction for bridge inspection imagery. The repository implements feature-based image registration,experimental geometry-aware stitching approaches using planar projection and triangular meshes and geometry-based experiments: Point-cloud plane fitting and camera-projection-based stitching onto a reference plane.

Tech Stack
Python · OpenCV · NumPy · SciPy · Matplotlib · Open3D · PyTorch / Segment Anything (for optional contour extraction)

Installation

git clone https://github.com/XiangjunMeng/stitch-images.git

cd stitch-images

pip install opencv-python numpy scipy matplotlib pillow tqdm

For optional geometry and contour extraction scripts, also install open3d, torch, and Meta's segment-anything package, and obtain a compatible SAM checkpoint.

Usage

Multi-image stitching: Edit image_dir in result.py to point to a folder of overlapping images, then run:

python result.py

The script matches overlapping images, estimates projective transformations, and displays a grayscale panorama.

Mesh-based workflow (experimental): Set the image paths and SAM checkpoint in the corresponding scripts, then run:

python extract_contours.py
python triangule_mesh.py
python stitch1.py

This workflow produces contour and mesh files and attempts a stitched output in stitched_results/.

Example Output
<img width="1920" height="967" alt="image" src="https://github.com/user-attachments/assets/fa8c755a-63ff-45ef-9cdb-54e049af77ae" />



