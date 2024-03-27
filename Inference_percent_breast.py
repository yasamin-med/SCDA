import os
import random
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import json
import argparse
import time
import shutil

def list_of_strings(arg):
    return arg.split(',')
def list_of_ints(arg):
    return list(map(float, arg.split(',')))

parser = argparse.ArgumentParser(description='Train Dataset on various nets on medical datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', type=str,
                    default='./data/', help='Path of trained model ')

parser.add_argument('--adjective_list', type=list_of_strings , default= [] , help ='list of adjectives')
parser.add_argument('--modes', type=list_of_strings , default= [] , help ='name of classes')
parser.add_argument('--percent_list', type=list_of_ints , default= [] , help ='percentage list')
parser.add_argument('--existing_images_base_directory', type=str,
                    default='./data/', help='Path of dataset for train')
parser.add_argument('--save_dir', type=str,
                    default='Augmented', help='directory to save new datasets')
parser.add_argument('--copy_flag', type=int,
                    default=1, help='copy base directory or not')
parser.add_argument('--prompt_structure', type=str, default= "an ultrasound photo of {class_name} tumor in breast", help= "structure of prompt")
args = parser.parse_args()
print(args)







# Initialize the model and pipeline
#model_path = "original"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)
pipe.unet.load_attn_procs(args.model_path)
pipe.to("cuda")
pipe.safety_checker = None

# Define the modes (classes) and the adjective list


#adjective_list = [ "colorful", "stylized", "high-contrast", "low-contrast", "posterized", "solarized", "sheared", "bright", "dark"]


# Base directory where current class images are stored


# Function to count images in a directory
def count_images_in_directory(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])


prompt_data = []

# Process images for each epoch
for percent in args.percent_list:
    # Choose a random adjective for this epoch
    for adjective in args.adjective_list:

        for mode in args.modes:
            # Path to the specific class directory
            
            class_directory = os.path.join(args.existing_images_base_directory, mode)

            # Count the number of images in the class directory
            existing_image_count = count_images_in_directory(class_directory)

            # Calculate the number of images to generate (half of existing count)
            image_count = int(existing_image_count * percent)
            if adjective == "":
                epoch_save_dir = f"/Augmented_{str(percent)}_percent/no_adjective/train/{mode}"
                epoch_save_dir = args.save_dir + epoch_save_dir
            # Directory for saving generated images for this epoch
            else:
                epoch_save_dir = f"/Augmented_{str(percent)}_percent/{adjective}/train/{mode}"
                epoch_save_dir = args.save_dir + epoch_save_dir

            # Ensure the directory exists
            if not os.path.exists(epoch_save_dir):
                os.makedirs(epoch_save_dir)
                
            files=os.listdir(class_directory)
            # iterating over all the files in 
            # the source directory
            if args.copy_flag == 1:
                for fname in files:
                    
                    # copying the files to the 
                    # destination directory
                    shutil.copy2(os.path.join(class_directory,fname), epoch_save_dir)
                print(f"copy done for {adjective} with {percent} percent")
                # Image generation logic
            
            for i in range(image_count):
                if mode == "normal":
                    basic_prompt = args.prompt_structure.replace("{class_name}" , "no")
                else:
                    
                    basic_prompt = args.prompt_structure.replace("{class_name}" , mode)

                #prompt = f"{adjective} ultrasound photo of {mode} tumor in breast" if adjective else f"ultrasound photo of a {mode} tumor in breast"
                prompt = f"{adjective}" + basic_prompt if adjective else basic_prompt
                image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
                name_save = os.path.join(epoch_save_dir, f"{i}_synth.png")
                image.save(name_save)

                # Log information for the last image of each epoch
                prompt_data.append({
                    "mode": mode,
                    "prompt": prompt,
                    "file_name": name_save,
                    "image_count": image_count,
                    "image location": epoch_save_dir
                })

        print("Updated folder.")
