import cv2
import numpy as np

def load_feat(path):
    kps = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                x, y, scale, angle = map(float, line.strip().split()[:2])
                kps.append(cv2.KeyPoint(x, y, scale, angle))
    return kps

def load_desc(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    dim, count = map(int, lines[0].strip().split())
    desc = [list(map(float, line.strip().split())) for line in lines[1:] if line.strip()]
    return np.array(desc, dtype = np.float32)

def load_matches(path):
    matches = [] 
    with open(path, 'r') as f:
        for line in f:
            if line.stripo():
                idx1, idx2 = map(int, line.strip().split())
                matches.append(cv2.DMatch(_queryIdx = idx1, _trainIdx = idx2, _imgIdx = 0, _distance = 0))
    return matches

