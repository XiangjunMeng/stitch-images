import os
from PIL import Image

root_dir = "c:/Users/robot/Desktop/bridgeModels/00002/bridgeModel/pictureset_0/in/nvm/views/"
output_dir = 'c:/Users/robot/Desktop/summer3d/image_dataset/'
image_paths = []
count = 0

for subfolder in sorted(os.listdir(root_dir)):
    subfolder_path = os.path.join(root_dir, subfolder)
    if os.path.isdir(subfolder_path):
        for file in os.listdir(subfolder_path):
            if file.lower().endswith(('.PNG', '.png')):
                image_path = os.path.join(subfolder_path, file)
                img = Image.open(image_path)
                img.save(os.path.join(output_dir, f"{count}.png"))
                image_paths.append(image_path)
                count += 1
                break
images = [Image.open(path) for path in image_paths]
print(f"total {len(images)} iamges")