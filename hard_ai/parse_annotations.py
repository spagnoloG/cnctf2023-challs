#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
import shutil
from tqdm import tqdm

annotations_dir = "/home/spagnologasper/Downloads/ILSVRC/Annotations/CLS-LOC/val"
val_samples_dir = "/home/spagnologasper/Downloads/ILSVRC/Data/CLS-LOC/val"
dataset_dir = "./dataset_train"
dataset_val_dir = "./dataset_val"


def grab_train_samples():
    train_samples = []
    for file in os.listdir(dataset_dir):
        train_samples.append(file)
    return train_samples


def return_validation_samples(train_samples: list):
    val_samples = []
    for file in os.listdir(annotations_dir):
        tree = ET.parse(os.path.join(annotations_dir, file))
        root = tree.getroot()
        for obj in root.iter("object"):
            for name in obj.iter("name"):
                if name.text in train_samples:
                    val_samples.append(file)

    return val_samples


def copy_over_files(val_samples: list):
    os.makedirs(dataset_val_dir, exist_ok=True)
    for file in tqdm(val_samples):
        shutil.copy(os.path.join(val_samples_dir, file[:-4] + ".JPEG"), dataset_val_dir)
        shutil.copy(os.path.join(annotations_dir, file), dataset_val_dir)


def main():
    train_samples = grab_train_samples()
    val_samples = return_validation_samples(train_samples)
    copy_over_files(val_samples)


if __name__ == "__main__":
    main()
