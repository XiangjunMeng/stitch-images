import os
import cv2
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import matplotlib.pyplot as plt

image_dir ="c:/Users/robot/Desktop/bridgeModels/00002/bridgeModel/pictureset_0/in"
output_dir = "./contours"
vis_dir = "./vis"
os.makedirs(output_dir, exist_ok=True)

sam_checkpoint = "c:/Users/robot/Downloads/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device = device)
predictor = SamPredictor(sam)

for filename in os.listdir(image_dir):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_dir, filename)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    H, W = image.shape[:2]
    input_box = np.array([0, 0, W, H])

    masks, _, _ = predictor.predict(box=input_box[None, :], multimask_output = True)
    mask = masks[0]

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"no contour found in {filename}")
        continue
    contour = max(contours, key=cv2.contourArea)
    contour = contour[:, 0, :]

    save_name = os.path.splitext(filename)[0] + "_contour.npy"
    np.save(os.path.join(output_dir, save_name), contour)

    print(f"saved contour for {filename} with {contour.shape[0]} points.")

    vis_image = image_rgb.copy()
    cv2.drawContours(vis_image, [contour.reshape(-1, 1, 2)], -1, (255, 0, 0), 2)
    cv2.circle(vis_image, tuple(np.mean(contour, axis=0).astype(int)), 5, (0, 255, 0), -1)

    vis_path = os.path.join(vis_dir, filename)
    cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))