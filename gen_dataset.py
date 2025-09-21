import os
import numpy as np
from PIL import Image
from resizeandpad import ResizeAndPad
from tqdm import tqdm

#Applies Resize and Pad to all images in the directory and stores the image 


ouput_dir = r"dataset\modified"
modified_raw = r"dataset\modified\RAW"
modifired_ref = r"dataset\modified\REF"
os.makedirs(ouput_dir,exist_ok = True)
os.makedirs(modified_raw,exist_ok = True)
os.makedirs(modifired_ref,exist_ok = True)

raw_dir = r"dataset\RAW"
ref_dir = r"dataset\REF"

raw_paths = sorted([os.path.join(raw_dir,fname) for fname in os.listdir(raw_dir) if fname.endswith((".jpg",".png"))])
ref_paths = sorted([os.path.join(ref_dir,fname) for fname in os.listdir(ref_dir) if fname.endswith((".jpg",".png"))])
transforms = ResizeAndPad((256,256))
i = 0
for raw,ref in tqdm(zip(raw_paths,ref_paths),total=len(raw_paths)):
    new_raw = Image.open(raw)
    new_ref = Image.open(ref)
    new_raw = transforms(new_raw)
    new_ref = transforms(new_ref)
    save_path_raw = os.path.join(modified_raw,f"{i}_img.png")
    save_path_ref = os.path.join(modifired_ref,f"{i}_img.png")

    new_ref.save(save_path_ref)
    new_raw.save(save_path_raw)
    i+=1


