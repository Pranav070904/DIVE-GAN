import os
import numpy as np
from PIL import Image
from resizeandpad import ResizeAndPad
from tqdm import tqdm
import random

#Applies Resize and Pad to all images in the directory and stores the image 

def apply_green_filter(img):
    """Applieas a Green Layer overlay to a PIL image"""
    green_colours = [
    (0, 180, 120),   # Blue-green
    (20, 200, 150),  # Teal-green  
    (0, 220, 100),   # Pure green but softer
    ]

    gc = random.choice(green_colours)

    green_layer = Image.new('RGB',img.size,gc)
    alpha = random.uniform(0.15, 0.35)
    return Image.blend(img,green_layer,alpha= alpha)


augment = False
augment_prob = .4

output_dir = None
modified_ref = None
modified_raw = None

if augment:
    output_dir = r"dataset\augmented"
    modified_raw = r"dataset\augmented\RAW"
    modified_ref = r"dataset\augmented\REF"
else:
    output_dir = r"dataset\EUVP\modified"
    modified_raw = r"dataset\EUVP\modified\RAW"
    modified_ref = r"dataset\EUVP\modified\REF"


os.makedirs(output_dir,exist_ok = True)
os.makedirs(modified_raw,exist_ok = True)
os.makedirs(modified_ref,exist_ok = True)

raw_dir = r"dataset\EUVP\RAW"
ref_dir = r"dataset\EUVP\REF"

augmented_count = 0

raw_paths = sorted([os.path.join(raw_dir,fname) for fname in os.listdir(raw_dir) if fname.endswith((".jpg",".png"))])
ref_paths = sorted([os.path.join(ref_dir,fname) for fname in os.listdir(ref_dir) if fname.endswith((".jpg",".png"))])
transforms = ResizeAndPad((256,256))
i = 0
for raw,ref in tqdm(zip(raw_paths,ref_paths),total=len(raw_paths)):
    new_raw = Image.open(raw)
    new_ref = Image.open(ref)
    if(augment and random.random()<augment_prob):
        new_raw = apply_green_filter(new_raw)
        augmented_count += 1
    new_raw = transforms(new_raw)
    new_ref = transforms(new_ref)
    save_path_raw = os.path.join(modified_raw,f"{i}_img.png")
    save_path_ref = os.path.join(modified_ref,f"{i}_img.png")

    new_ref.save(save_path_ref)
    new_raw.save(save_path_raw)
    i+=1


print("Dataset Generated")
if(augment):
    print(f"{augmented_count} images augmented.")

