from datasets import Dataset
from datasets import load_dataset, Image,load_from_disk
import os
from pathlib import Path
import torch
import cv2



path_image = []
text_image = []
root_train_dataset = "/home/yasamin/Documents/Covid Dataset/Original/split0"
root_main = root_train_dataset + "/covid"


count = 0
for root, dirs, files in os.walk(root_main):
    for name in files:
        path_image.append((os.path.join( root , name)))
        prompt = "an ultrasound photo of covid lung"
        
        temp = Image((os.path.join( root , name)))
        count = count + 1
        
        text_image.append(prompt)

root_main = root_train_dataset + "/pneumonia"


for root, dirs, files in os.walk(root_main):
    for name in files:
        path_image.append((os.path.join( root , name)))
        prompt = "an ultrasound photo of pneumonia lung"
        
        temp = Image((os.path.join( root , name)))
        count = count + 1
        
        text_image.append(prompt)

root_main = root_train_dataset + "/regular"


for root, dirs, files in os.walk(root_main):
    for name in files:
        path_image.append((os.path.join( root , name)))
        prompt = "an ultrasound photo of regular lung"
        
        temp = Image((os.path.join( root , name)))
        count = count + 1
        
        text_image.append(prompt)

print(len(text_image))
print(len(path_image))
dataset = Dataset.from_dict({"image": path_image , "text": text_image}).cast_column("image", Image())

dataset.push_to_hub("yasimed/Covid_dataset", private=True)


dataset = load_dataset("yasimed/Covid_dataset",use_auth_token="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")
print(dataset)
