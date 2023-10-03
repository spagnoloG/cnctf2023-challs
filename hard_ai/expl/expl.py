import requests
from bs4 import BeautifulSoup
from time import sleep
import torch
import timm
from PIL import Image
import torchvision.transforms.functional as F
import os
from torch.utils.data import Dataset

BASE_URL = "http://localhost:5000/"  # Replace with the actual website domain


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
                self.file_names.append(
                    os.path.join(self.dataset_path, label, file_name)
                )

    def _create_label_map(self):
        unique_labels = sorted(list(set(self.labels)))
        self.label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

    def return_label_map(self):
        return self.label_to_int

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        label = file_name.split("/")[-2]
        image = Image.open(file_name).convert("RGB")
        image = F.to_tensor(image)
        image = F.resize(image, (224, 224), antialias=True)
        image = F.normalize(
            image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )  # ImageNet mean and std

        return image, self.label_to_int[label]

    def __len__(self):
        return len(self.file_names)


def load_model(path_to_checkpoint, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("res2next50", pretrained=False, num_classes=num_classes)
    model = model.to(device)

    checkpoint = torch.load(path_to_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # Set model to evaluation mode

    return model


def classify_image(model, image_path, label_map):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = F.to_tensor(image)
    image = F.resize(image, (224, 224), antialias=True)
    image = F.normalize(
        image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    )  # ImageNet mean and std
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(image)
        _, predicted_class = torch.max(outputs, 1)

    return [key for key, value in label_map.items() if value == predicted_class.item()][
        0
    ]  # Return the class label


# Using a session to maintain state across requests
with requests.Session() as session:
    label_map = ImageNetSubsetTrain().return_label_map()

    model = load_model("./checkpoint.pth", len(label_map))

    for i in range(100):
        # Request the page
        response = session.get(BASE_URL)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        img_tag = soup.find("img", class_="shadow-2")

        p_tags = soup.find_all("p")
        print(p_tags)

        # Download image
        img_url = BASE_URL + img_tag["src"]
        img_response = session.get(img_url, stream=True)
        img_response.raise_for_status()

        with open(f"image_{i}.jpg", "wb") as img_file:
            for chunk in img_response.iter_content(chunk_size=8192):
                img_file.write(chunk)

        classified_class = classify_image(model, f"image_{i}.jpg", label_map)

        # Assume you know the class or have a way to predict it.
        # Replace 'your_class_here' with the actual class.
        data = {"user_input": classified_class}

        # Submit the class
        submit_response = session.post(BASE_URL + "/submit", data=data)
        submit_response.raise_for_status()

        if "cnctf" in submit_response.text:
            print(submit_response.text)
            break

        sleep(0.1)

print("Finished all 100 iterations!")
