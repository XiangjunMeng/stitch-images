import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []
images.append(cv2.imread("c:/Users/robot/Desktop/bridgeModels/00004/pictureset_0/in/image000000.jpg"))
images.append(cv2.imread("c:/Users/robot/Desktop/bridgeModels/00004/pictureset_0/in/image000001.jpg"))

sift = cv2.SIFT_create()
keypoints = []
descriptors = []
for image in images:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    keypoints.append(kp)
    descriptors.append(des)

FLANN_INDEX_KETREEE = 1
index_params = dict(algorithm = FLANN_INDEX_KETREEE, trees = 5)
search_params = dict(checks = 50)
matcher = cv2.FlannBasedMatcher(index_params, search_params)
matches = []
for i in range(len(descriptors)-1):
    matches.append(matcher.knnMatch(descriptors[i], descriptors[i+1], k = 2))

good_matches = []
for match in matches:
    good = []
    for m, n in match:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    good_matches.append(good)

homographies = []
for i, match in enumerate(good_matches):
    if len(match) >= 4:
        src_pts = np.float32([keypoints[i][m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[i+1][m.trainIdx].pt for m in match]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 2.0)
        homographies.append(H)

result = images[0]
for i in range(len(images)-1):
    result = cv2.warpPerspective(result, homographies[i], (result.shape[1] + images[i+1].shape[1], result.shape[0]))
    result[0:images[i+1].shape[0], 0:images[i+1].shape[1]] = images[i+1]

plt.figure(figsize=(12, 8))
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title("stitched result")
plt.axis("off")
plt.tight_layout()
plt.show()