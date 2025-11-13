
import os
import pandas as pd

from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

###################################################

def load_synset_mapping(synset_mapping_file):
    """Load synset mapping (WordNet ID → Class Index) from the given file."""
    mapping = {}
    with open(synset_mapping_file, 'r') as f:
        for idx, line in enumerate(f.readlines()):
            wnid = line.strip().split(" ")[0]  # Extract only the WordNet ID
            mapping[wnid] = idx  # Assign index (0-999)

    # Debugging: Print first few mappings
    print(f"✅ Loaded {len(mapping)} synset mappings")
    print("🔍 First 5 mappings:", list(mapping.items())[:5])

    return mapping


def preprocess_image(image_path, preprocess):
    """Preprocess an image for the model."""
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)  # Add batch dimension


def predict(image_path, model, preprocess, synset_mapping):
    """Make a prediction on a single image."""
    try:
        input_tensor = preprocess_image(image_path, preprocess)
        with torch.no_grad():
            output = model(input_tensor)
        probabilities = F.softmax(output[0], dim=0)
        top_class = probabilities.argmax().item()
        return synset_mapping[top_class]
    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None


def load_ground_truth(val_solution_file):
    """Load ground truth labels for classification."""
    image_ids = []
    ground_truths = []
    with open(val_solution_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            image_ids.append(parts[0])  # ImageId
            ground_truths.append(parts[1].split()[0])  # SynsetID (first part of PredictionString)

    return image_ids, ground_truths


###################################################

class ImageNetValDataset(Dataset):
    def __init__(self, img_dir, label_file, synset_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        # Load synset mapping (WordNet ID → Class Index)
        self.synset_mapping = load_synset_mapping(synset_file)

        # Load the CSV with ground truth labels
        self.labels = pd.read_csv(label_file, header=0, usecols=[0, 1], names=["filename", "label"], delimiter=",")

        # Ensure filenames are properly formatted
        self.labels["filename"] = self.labels["filename"].astype(str).str.strip()
        self.labels["label"] = self.labels["label"].astype(str).str.strip()

        # Remove bounding box information (only keep first word of label)
        self.labels["label"] = self.labels["label"].apply(lambda x: x.split(" ")[0])

        # Append `.JPEG` if missing in filenames
        self.labels["filename"] = self.labels["filename"].apply(lambda x: x + ".JPEG" if not x.endswith(".JPEG") else x)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0]
        wnid_label = self.labels.iloc[idx, 1]  # WordNet ID

        # Convert WordNet ID to class index
        if wnid_label in self.synset_mapping:
            label = self.synset_mapping[wnid_label]
        else:
            raise ValueError(f"❌ Label {wnid_label} not found in synset mapping!")

        img_path = os.path.join(self.img_dir, img_name)

        # Debugging: Print full path before loading
        # print(f"🔍 Checking file: {img_path}")

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"❌ Image file not found: {img_path}")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        else:
            raise ValueError("❌ No transform function provided!")

            # Convert label to a PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)  # Ensure label is a tensor

        return image, label


###################################################

def getImageNet(datasetdir, size=256, batchsize=64):

    val_img_dir = os.path.join(datasetdir, 'ImageNet', 'data', 'val')
    label_file = os.path.join(datasetdir, 'ImageNet', 'data', 'LOC_val_solution.csv')
    synset_file = os.path.join(datasetdir, 'ImageNet', 'data', 'LOC_synset_mapping.txt')

    transf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure synset file is passed
    # print(f'-I({__file__}): Loading ImageNet dataset from {val_img_dir}')

    test_dataset = ImageNetValDataset(val_img_dir, label_file, synset_file, transform=transf)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batchsize,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=torch.cuda.is_available())

    print(f'-I({__file__}): ImageNet loaded')

    return test_loader


