import os
import shutil
import random

import json
import requests

import urllib.request

url = (
    "https://raw.githubusercontent.com/tensorflow/models/master/"
    "research/slim/datasets/imagenet_lsvrc_2015_synsets.txt"
)

response = urllib.request.urlopen(url)
synset_list = [line.strip().decode('utf-8') for line in response.readlines()]
synset_to_idx = {syn: idx for idx, syn in enumerate(synset_list)}

selected_classes = [
    # Animals (vertebrates & invertebrates)
    'n01440764',  # tench (fish)
    'n01518878',  # ostrich
    'n01530575',  # brambling
    'n01632777',  # axolotl
    'n02098105', # soft-coated wheaten terrier
    'n12620546', #hip
    'n03598930',#jigsaw puzzle'
    'n03599486', #jinrikisha

    
    # Vehicles & Transport
    'n03662601',  # lifeboat
    'n03788365', #mosquito net
    'n03814906', #necklace
    'n03857828', #oscilloscope
    'n03424325', #gasmask
    'n03444034', #go-kart
    'n04044716', #radio telescope
    'n04146614', #schoolbus
    'n04266014', #spaceshuttle
    'n07697313', #prezel
    'n07715103', #cauliflower
    'n02102040', #English springer
    
]

#subset_to_model_label = {syn: synset_to_idx[syn] for syn in selected_classes}
#print(subset_to_model_label)
# 5. Print or save
#print(json.dumps(subset_to_model_label, indent=2))
# Paths
src_dir = '/data/datasets/imagenet/val'
dst_dir = "data/imagenet_subset"
samples_per_class = 50

# Create output root
os.makedirs(dst_dir, exist_ok=True)

for classes in selected_classes:
    class_path = os.path.join(src_dir, classes)
    if not os.path.isdir(class_path):
        print(f"Warning: {classes} not found in source directory.")
        continue

    images = os.listdir(class_path)
    if len(images) < samples_per_class:
        print(f"Warning: Class {classes} has only {len(images)} images.")
        continue

    random.shuffle(images)
    selected_images = images[:samples_per_class]

    dst_class_path = os.path.join(dst_dir, str(classes))
    os.makedirs(dst_class_path, exist_ok=True)

    for img in selected_images:
        src_img = os.path.join(class_path, img)
        dst_img = os.path.join(dst_class_path, img)
        shutil.copyfile(src_img, dst_img)

print("Subset saved using numeric class labels.")