from datasets import Dataset
from datasets import load_dataset, Image,load_from_disk
import os
from pathlib import Path
import torch
import cv2
import argparse

def list_of_strings(arg):
    return arg.split(',')
parser = argparse.ArgumentParser(description='Preparing Datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--train_dir', type=str, default="", help= "directory of train dataset")
parser.add_argument('--classes', type= list_of_strings, default=['benign','malignant','normal'], help= "name of classes with camma and no space")
parser.add_argument('--prompt_structure', type=str, default= "an ultrasound photo of {class_name} tumor in breast", help= "structure of prompt")
parser.add_argument('--dataset_path', type=str, default= "yasimed/split_dataset", help= "path of dataset in hugging face")
parser.add_argument('--token', type=str, default= "", help= "token of access")
args = parser.parse_args()
print(args)


path_image = []
text_image = []



count = 0
for class_name in args.classes:
    root_main = os.path.join(args.train_dir , class_name)
    
    for root, dirs, files in os.walk(root_main):
        for name in files:
            path_image.append((os.path.join( root , name)))
            if class_name == "normal":
                prompt = args.prompt_structure.replace("{class_name}" , "no")
            prompt = args.prompt_structure.replace("{class_name}" , class_name)
            
            temp = Image((os.path.join( root , name)))
            count = count + 1
            
            text_image.append(prompt)

print("Number of Prompts: ", len(text_image))
print("Number of Images: ", len(path_image))
dataset = Dataset.from_dict({"image": path_image , "text": text_image}).cast_column("image", Image())

dataset.push_to_hub(args.dataset_path, private=True)


dataset = load_dataset(args.dataset_path,use_auth_token=args.token)

print(dataset)
