#!/usr/bin/env python3

import os
from PIL import Image
import torchvision.transforms.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
import timm
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

class ImageNetSubsetTrain(Dataset):
    def __init__(self):
        self.dataset_path = "../dataset_train/"
        self.file_names = []
        self.labels = []
        self.label_to_int = {}
        self._load_data()
        self._create_label_map()

    def _load_data(self):
        for label in os.listdir(self.dataset_path):
            if os.path.isdir(os.path.join(self.dataset_path, label)):
                self.labels.append(label)

        for label in self.labels:
            for file_name in os.listdir(os.path.join(self.dataset_path, label)):
                self.file_names.append(os.path.join(self.dataset_path, label, file_name))

    def _create_label_map(self):
        unique_labels = sorted(list(set(self.labels)))
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    def _one_hot_encode(self, label):
        one_hot = torch.zeros(len(self.label_to_int))
        one_hot[self.label_to_int[label]] = 1
        return one_hot

    def return_label_map(self):
        lmap = {}
        for label, idx in self.label_to_int.items():
            one_hot = self._one_hot_encode(label)
            lmap[label] = one_hot
        return lmap

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = file_name.split('/')[-2]
        image = Image.open(file_name).convert('RGB')
        image = F.to_tensor(image)
        image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean and std
        image = F.resize(image, (224, 224), antialias=True)

        return image, self._one_hot_encode(label)

    def __len__(self):
        return len(self.file_names)


class ImageNetSubsetVal(Dataset):
    def __init__(self, label_to_hvec):
        self.dataset_path = "../dataset_val/"
        self.labels_to_file = []
        self.label_to_hvec = label_to_hvec
        self._load_data()

    def _load_data(self):
        for file in os.listdir(self.dataset_path):
            if file.endswith(".xml"):
                tree = ET.parse(os.path.join(self.dataset_path, file))
                root = tree.getroot()
                for obj in root.iter('object'):
                    for name in obj.iter('name'):
                        self.labels_to_file.append((name.text, file))

    def __getitem__(self, idx):
        label, file = self.labels_to_file[idx]
        image_name = file.split('.')[0] + '.JPEG'
        image = Image.open(os.path.join(self.dataset_path, image_name)).convert('RGB')
        image = F.to_tensor(image)
        image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet mean and std
        image = F.resize(image, (224, 224), antialias=True)

        return image, self.label_to_hvec[label]


    def __len__(self):
        return len(self.labels_to_file)



def train(dataloader_train, dataloader_val, epochs=4):
    num_classes = len(dataloader_train.dataset.label_to_int)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    for epoch in range(epochs):
        # Training Loop
        model.train()
        train_loss = 0.0
        for images, labels in tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.argmax(1).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)

        train_loss = train_loss / len(dataloader_train.dataset)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader_val, desc=f"Epoch {epoch+1}/{epochs}"):
                images, labels = images.to(device), labels.argmax(1).to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss = val_loss / len(dataloader_val.dataset)
        val_acc = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Update learning rate
        scheduler.step()

    print('Finished Training!') 


def main():
    dataset_train = ImageNetSubsetTrain()
    label_to_hvec = dataset_train.return_label_map()
    dataset_val = ImageNetSubsetVal(label_to_hvec)

    dataloader_train = DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)
    dataloader_val = DataLoader(dataset_val, batch_size=16, shuffle=False, num_workers=4)
    torch.backends.cudnn.benchmark = True # Enable faster training

    train(dataloader_train, dataloader_val)

if __name__ == "__main__":
    main()
