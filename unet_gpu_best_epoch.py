import warnings
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import rasterio
import cv2
from torch.utils.data import DataLoader, Dataset
from rasterio.enums import Resampling
import shutil

warnings.filterwarnings("ignore")

# Define GPU or CPU device context
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on device: {device}")

country = "thailand"
binary_dir = f"Data/label_{country}"
sentinel_dir = f"Data/sentinel_{country}"
target_sentinel_dir  = f"Data/sentinel_patches_{country}"

EPOCH_NUM = 30

# Directory paths
train_prefix = 'Data/testing/train/'
val_prefix = 'Data/testing/val/'
train_label_prefix = 'Data/testing/train/label'
train_sentinel_prefix = 'Data/testing/train/sentinel'
val_label_prefix = 'Data/testing/val/label'
val_sentinel_prefix = 'Data/testing/val/sentinel'


# Create directories
os.makedirs(train_prefix, exist_ok=True)
os.makedirs(val_prefix, exist_ok=True)
os.makedirs(train_label_prefix, exist_ok=True)
os.makedirs(train_sentinel_prefix, exist_ok=True)
os.makedirs(val_label_prefix, exist_ok=True)
os.makedirs(val_sentinel_prefix, exist_ok=True)


# List all files in the directories
def list_files(dir_path):
    return [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

binary_files = list_files(binary_dir)
sentinel_files = list_files(sentinel_dir)


def extract_id(filename):
    return filename.split('.')[0]


# Extract unique IDs from both directories
binary_ids = set(extract_id(f) for f in binary_files)
sentinel_ids = set(extract_id(f) for f in sentinel_files)

# Combine and split unique IDs into training and validation sets
all_ids = list(binary_ids.union(sentinel_ids))
random.shuffle(all_ids)
split_index = int(0.8 * len(all_ids))

train_ids = all_ids[:split_index]
val_ids = all_ids[split_index:]

import shutil

def copy_and_rename_files(ids, src_dir, dest_dir, prefix):
    for uid in ids:
        for file in list_files(src_dir):
            if extract_id(file) == uid:
                src_file = os.path.join(src_dir, file)
                new_filename = f"{prefix}_{uid}.tif"
                dest_file = os.path.join(dest_dir, new_filename)
                shutil.copy(src_file, dest_file)

# Process and copy binary files
#copy_and_rename_files(train_ids, binary_dir, train_label_prefix, 'label')
#copy_and_rename_files(val_ids, binary_dir, val_label_prefix, 'label')
#copy_and_rename_files(train_ids, sentinel_dir, train_sentinel_prefix, 's2')
#copy_and_rename_files(val_ids, sentinel_dir, val_sentinel_prefix, 's2')

print(f"Copied and renamed files for {len(train_ids)} training and {len(val_ids)} validation IDs.")

# Function to count files in a directory
def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])

# Print the number of files in each directory
print("Number of files in train_label:", count_files(train_label_prefix))
print("Number of files in train_sentinel:", count_files(train_sentinel_prefix))
print("Number of files in val_label:", count_files(val_label_prefix))
print("Number of files in val_sentinel:", count_files(val_sentinel_prefix))

# Helper function to load TIFF images
def load_tiff_image(file_path):
    with rasterio.open(file_path) as src:
        return src.read().transpose(1, 2, 0)

# Data Preprocessing
class SegmentationDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size):
        self.images = sorted([os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.tif')])
        self.masks = sorted([os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.endswith('.tif')])
        self.image_size = image_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = load_tiff_image(self.images[idx])
        mask = load_tiff_image(self.masks[idx])

        image = cv2.resize(image, (self.image_size, self.image_size))
        if image.shape[-1] != 10:
            image = np.repeat(image[..., np.newaxis], 10, axis=-1)

        mask = cv2.resize(mask, (self.image_size, self.image_size))
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=-1)

        # Ensure the correct shape for the model input/output
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).squeeze(-1)  # Remove the extra dimension

        return image, mask

# Define U-Net Model in PyTorch
class UNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512)
        )
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Conv2d(64, output_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.upconv(x)
        x = self.out_conv(x)
        return torch.sigmoid(x)

# Define an accuracy calculation function
def calculate_accuracy(outputs, masks):
    # Apply a threshold to get binary predictions
    predictions = (outputs > 0.5).float()
    correct = (predictions == masks).float().sum()
    accuracy = correct / masks.numel()
    return accuracy.item()

# Training Setup
def train_model(model, train_loader, val_loader, epochs=EPOCH_NUM):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    model.to(device)

    best_val_accuracy = 0  # Initialize best validation accuracy

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_accuracy = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += calculate_accuracy(outputs, masks)

        # Average training metrics for the epoch
        avg_train_loss = train_loss / len(train_loader)
        avg_train_accuracy = train_accuracy / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_train_loss:.4f}, Accuracy: {avg_train_accuracy:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_accuracy = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_accuracy += calculate_accuracy(outputs, masks)

        # Average validation metrics for the epoch
        avg_val_loss = val_loss / len(val_loader)
        avg_val_accuracy = val_accuracy / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {avg_val_accuracy:.4f}")

        # Check if the current model has the best validation accuracy
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            # Save the model if it achieves a new best validation accuracy
            model_save_path = f'Data/my_unet_model_{country}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch+1} with validation accuracy: {best_val_accuracy:.4f}")

# Instantiate model and data loaders
train_dataset = SegmentationDataset(train_sentinel_prefix, train_label_prefix, 512)
val_dataset = SegmentationDataset(val_sentinel_prefix, val_label_prefix, 512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")

model = UNet(input_channels=10, output_channels=1)
train_model(model, train_loader, val_loader)

import os
import torch
import rasterio
import numpy as np
from rasterio.enums import Resampling

def run_inference_and_save_tiff(predictions_folder=f'Data/inference_{country}', input_folder= sentinel_dir):
    os.makedirs(predictions_folder, exist_ok=True)

    # Load the model weights and set to evaluation mode
    model.load_state_dict(torch.load(f'Data/my_unet_model_{country}.pth'))
    model.to(device)
    model.eval()

    for image_file in os.listdir(input_folder):
        if image_file.endswith('.tif'):
            image_path = os.path.join(input_folder, image_file)
            with rasterio.open(image_path) as src:
                original_metadata = src.meta.copy()

                # Read image and normalize it if needed (e.g., divide by 255.0 if in [0, 255] range)
                image_array = src.read(out_shape=(src.count, src.height, src.width), resampling=Resampling.nearest)
                image_array = np.moveaxis(image_array, 0, -1).astype(np.float32) / 255.0

                # Convert image to tensor and add batch dimension
                image_tensor = torch.tensor(image_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

                # Run model inference
                with torch.no_grad():
                    # Get the raw output from the model
                    raw_prediction = model(image_tensor)
                    probability_map = raw_prediction.cpu().squeeze().numpy()

                # Ensure values are clipped to [0, 1]
                probability_map = np.clip(probability_map, 0, 1).astype(np.float32)

                # Update metadata for saving as a probability map
                original_metadata.update({
                    'count': 1, 'dtype': 'float32', 'driver': 'GTiff'
                })

                # Save the probability map
                output_path = os.path.join(predictions_folder, f'inference_{os.path.splitext(image_file)[0]}.tif')
                with rasterio.open(output_path, 'w', **original_metadata) as dst:
                    dst.write(probability_map, 1)

# Call the inference function
run_inference_and_save_tiff()
print("Inference and saving completed.")


import os
import numpy as np
import rasterio
import random

# Define the path to the inference folder
inference_folder = f'Data/inference_{country}'

# Get all .tif files in the folder
tif_files = [f for f in os.listdir(inference_folder) if f.endswith('.tif')]

# Randomly select 5 files from the list
random_files = random.sample(tif_files, min(5, len(tif_files)))

# Calculate and print statistics for each selected file
for file in random_files:
    file_path = os.path.join(inference_folder, file)

    with rasterio.open(file_path) as src:
        # Read the probability values
        data = src.read(1)  # Read the single channel

        # Calculate statistics
        min_val = np.min(data)
        max_val = np.max(data)
        median_val = np.median(data)

        print(f"File: {file}")
        print(f"  Min: {min_val}")
        print(f"  Max: {max_val}")
        print(f"  Median: {median_val}")
        print("\n")
