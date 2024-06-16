import seaborn as sns

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import glob
import json
import collections
import torch
import torch.nn as nn

import pydicom as dicom
import matplotlib.patches as patches

from matplotlib import animation, rc
import pandas as pd

import pydicom as dicom # dicom
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# read data
train_path = '/home/piljae/kaggle/rsna-2024-lumbar-spine-degenerative-classification/'

train  = pd.read_csv(train_path + 'train.csv')
label = pd.read_csv(train_path + 'train_label_coordinates.csv')
train_desc  = pd.read_csv(train_path + 'train_series_descriptions.csv')
test_desc   = pd.read_csv(train_path + 'test_series_descriptions.csv')
sub         = pd.read_csv(train_path + 'sample_submission.csv')



######################################################## Image Viewer ####################################################################

import pydicom
import matplotlib.pyplot as plt

# Function to generate image paths based on directory structure
def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        images = os.listdir(series_dir)
        image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths


# Function to open and display DICOM images
def display_dicom_images(image_paths):
    plt.figure(figsize=(15, 5))  # Adjust figure size if needed
    for i, path in enumerate(image_paths[:3]):
        ds = pydicom.dcmread(path)
        plt.subplot(1, 3, i+1)
        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
        plt.title(f"Image {i+1}")
        plt.axis('off')
    plt.show()

# Function to open and display DICOM images along with coordinates
def display_dicom_with_coordinates(image_paths, label_df):
    fig, axs = plt.subplots(1, len(image_paths), figsize=(18, 6))
    
    for idx, path in enumerate(image_paths):  # Display images
        study_id = int(path.split('/')[-3])
        series_id = int(path.split('/')[-2])
        
        # Filter label coordinates for the current study and series
        filtered_labels = label_df[(label_df['study_id'] == study_id) & (label_df['series_id'] == series_id)]
        
        # Read DICOM image
        ds = pydicom.dcmread(path)
        
        # Plot DICOM image
        axs[idx].imshow(ds.pixel_array, cmap='gray')
        axs[idx].set_title(f"Study ID: {study_id}, Series ID: {series_id}")
        axs[idx].axis('off')
        
        # Plot coordinates
        for _, row in filtered_labels.iterrows():
            axs[idx].plot(row['x'], row['y'], 'ro', markersize=5)
        
    plt.tight_layout()
    plt.show()

# Load DICOM files from a folder
def load_dicom_files(path_to_folder):
    files = [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith('.dcm')]
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[-1]))
    return files


# Generate image paths for train and test data
train_image_paths = generate_image_paths(train_desc, f'{train_path}/train_images')
test_image_paths = generate_image_paths(test_desc, f'{train_path}/test_images')


# Display the first three DICOM images
#display_dicom_images(train_image_paths)

# Display DICOM images with coordinates
study_id = "100206310"
study_folder = f'{train_path}/train_images/{study_id}'

image_paths = []
for series_folder in os.listdir(study_folder):
    series_folder_path = os.path.join(study_folder, series_folder)
    dicom_files = load_dicom_files(series_folder_path)
    if dicom_files:
        image_paths.append(dicom_files[0])  # Add the first image from each series

#display_dicom_with_coordinates(image_paths, label)

#################################################################################################################################


#################################################### Deata processing ###########################################################

# Define function to reshape a single row of the DataFrame
def reshape_row(row):
    data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}
    
    for column, value in row.items():
        if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
            parts = column.split('_')
            condition = ' '.join([word.capitalize() for word in parts[:-2]])
            level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
            data['study_id'].append(row['study_id'])
            data['condition'].append(condition)
            data['level'].append(level)
            data['severity'].append(value)
    
    return pd.DataFrame(data)


# Reshape the DataFrame for all rows
new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)

# Merge the dataframes on the common columns
merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')

# Merge the dataframes on the common column 'series_id'
final_merged_df = pd.merge(merged_df, train_desc, on='series_id', how='inner')

# Merge the dataframes on the common column 'series_id'
final_merged_df = pd.merge(merged_df, train_desc, on=['series_id','study_id'], how='inner')
# Display the first few rows of the final merged dataframe

# Create the row_id column
final_merged_df['row_id'] = (
    final_merged_df['study_id'].astype(str) + '_' +
    final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
    final_merged_df['level'].str.lower().str.replace('/', '_')
)

# Create the image_path column
final_merged_df['image_path'] = (
    f'{train_path}/train_images/' + 
    final_merged_df['study_id'].astype(str) + '/' +
    final_merged_df['series_id'].astype(str) + '/' +
    final_merged_df['instance_number'].astype(str) + '.dcm'
)

# Define the base path for test images
base_path = '/home/piljae/kaggle/rsna-2024-lumbar-spine-degenerative-classification/test_images/'

# Function to get image paths for a series
def get_image_paths(row):
    series_path = os.path.join(base_path, str(row['study_id']), str(row['series_id']))
    if os.path.exists(series_path):
        return [os.path.join(series_path, f) for f in os.listdir(series_path) if os.path.isfile(os.path.join(series_path, f))]
    return []

# Mapping of series_description to conditions
condition_mapping = {
    'Sagittal T1': {'left': 'left_neural_foraminal_narrowing', 'right': 'right_neural_foraminal_narrowing'},
    'Axial T2': {'left': 'left_subarticular_stenosis', 'right': 'right_subarticular_stenosis'},
    'Sagittal T2/STIR': 'spinal_canal_stenosis'
}

# Create a list to store the expanded rows
expanded_rows = []

# Expand the dataframe by adding new rows for each file path
for index, row in test_desc.iterrows():
    image_paths = get_image_paths(row)
    conditions = condition_mapping.get(row['series_description'], {})
    if isinstance(conditions, str):  # Single condition
        conditions = {'left': conditions, 'right': conditions}
    for side, condition in conditions.items():
        for image_path in image_paths:
            expanded_rows.append({
                'study_id': row['study_id'],
                'series_id': row['series_id'],
                'series_description': row['series_description'],
                'image_path': image_path,
                'condition': condition,
                'row_id': f"{row['study_id']}_{condition}"
            })

# Create a new dataframe from the expanded rows
expanded_test_desc = pd.DataFrame(expanded_rows)

# Train_data and test_data
train_data = final_merged_df
test_data = expanded_test_desc

import os

# Define a function to check if a path exists
def check_exists(path):
    return os.path.exists(path)

# Define a function to check if a study ID directory exists
def check_study_id(row):
    study_id = row['study_id']
    path = f'{train_path}/train_images/{study_id}'
    return check_exists(path)

# Define a function to check if a series ID directory exists
def check_series_id(row):
    study_id = row['study_id']
    series_id = row['series_id']
    path = f'{train_path}/train_images/{study_id}/{series_id}'
    return check_exists(path)

# Define a function to check if an image file exists
def check_image_exists(row):
    image_path = row['image_path']
    return check_exists(image_path)

# Apply the functions to the train_data dataframe
train_data['study_id_exists'] = train_data.apply(check_study_id, axis=1)
train_data['series_id_exists'] = train_data.apply(check_series_id, axis=1)
train_data['image_exists'] = train_data.apply(check_image_exists, axis=1)

# Filter train_data
train_data = train_data[(train_data['study_id_exists']) & (train_data['series_id_exists']) & (train_data['image_exists'])]
train_data = train_data.dropna()


# resampling

from sklearn.utils import resample

class_counts = train_data['severity'].value_counts()

# 최대 클래스 수
max_count = class_counts.max()

# 각 클래스의 데이터를 균형 맞추기 위해 리샘플링
balanced_data = pd.DataFrame()

for severity in class_counts.index:
    class_data = train_data[train_data['severity'] == severity]
    if len(class_data) < max_count:

        class_data = resample(class_data, replace=True, n_samples=max_count, random_state=42)
    balanced_data = pd.concat([balanced_data, class_data])

train_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)


####################################################### Loading Data ###########################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from PIL import Image
import cv2

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        label = self.dataframe['severity'][index]

        # image가 None인 경우 처리
        if image is None:
            image = np.zeros((256, 256), dtype=np.uint8)  # 또는 이미지 크기에 맞게 설정
        
        # image crop
        x = round(self.dataframe['x'][index])
        y = round(self.dataframe['y'][index])
        
        gap_x = round(image.shape[0] / 10)
        gap_y = round(image.shape[1] / 10)

        # 이미지 크롭 범위가 유효한지 확인
        if y-gap_y < 0 or y+gap_y > image.shape[0] or x-gap_x < 0 or x+gap_x > image.shape[1]:
            image = np.zeros((256, 256), dtype=np.uint8)  # 유효하지 않으면 검정색 이미지 반환
        else:
            image = image[y-gap_y : y+gap_y, x-gap_x : x+gap_x]

        image = cv2.equalizeHist(image)
        image = Image.fromarray(image).resize((224, 224), Image.BILINEAR)

        # Convert to 3 channels (RGB)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Define the transforms with rotation for training data
train_transforms = transforms.Compose([
    transforms.RandomRotation(45),  # Randomly rotate the image by 45 degrees
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the transforms for validation data
val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to create datasets and dataloaders for each series description
def create_datasets_and_loaders(df, series_description, batch_size=8, train_transform=None, val_transform=None):
    filtered_df = df[df['series_description'] == series_description]
    
    train_df, val_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = CustomDataset(train_df, transform=train_transform)
    val_dataset = CustomDataset(val_df, transform=val_transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, valloader, len(train_df), len(val_df)

# # Define the transforms
# transform = transforms.Compose([
#     transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
#     transforms.ToPILImage(),
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
# ])

# Create dataloaders for each series description
dataloaders = {}
lengths = {}

trainloader_t1, valloader_t1, len_train_t1, len_val_t1 = create_datasets_and_loaders(train_data, 'Sagittal T1', batch_size=8, train_transform=train_transforms, val_transform=val_transforms)
trainloader_t2, valloader_t2, len_train_t2, len_val_t2 = create_datasets_and_loaders(train_data, 'Axial T2', batch_size=8, train_transform=train_transforms, val_transform=val_transforms)
trainloader_t2stir, valloader_t2stir, len_train_t2stir, len_val_t2stir = create_datasets_and_loaders(train_data, 'Sagittal T2/STIR', batch_size=8, train_transform=train_transforms, val_transform=val_transforms)

dataloaders['Sagittal T1'] = (trainloader_t1, valloader_t1)
dataloaders['Axial T2'] = (trainloader_t2, valloader_t2)
dataloaders['Sagittal T2/STIR'] = (trainloader_t2stir, valloader_t2stir)

lengths['Sagittal T1'] = (len_train_t1, len_val_t1)
lengths['Axial T2'] = (len_train_t2, len_val_t2)
lengths['Sagittal T2/STIR'] = (len_train_t2stir, len_val_t2stir)

# # Dictionary mapping labels to indices
# label_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}

import matplotlib.pyplot as plt

# Function to visualize a batch of images
def visualize_batch(dataloader):
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, len(images), figsize=(20, 5))
    for i, (img, lbl) in enumerate(zip(images, labels)):
        ax = axes[i]
        img = img.permute(1, 2, 0)  # Convert to HWC for visualization
        ax.imshow(img)
        ax.set_title(f"Label: {lbl}")
        ax.axis('off')
    plt.show()

################################ Data Visulization ##############################################

# # Visualize samples from each dataloader
# print("Visualizing Sagittal T1 samples")
# visualize_batch(valloader_t1)
# print("Visualizing Axial T2 samples")
# visualize_batch(trainloader_t2)
# print("Visualizing Sagittal T2/STIR samples")
# visualize_batch(trainloader_t2stir)


################################ Model ###########################################################

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomResNet(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=None):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)  # Use pre-trained weights
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=0.5),  # Dropout to prevent overfitting
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    

# Path to the locally uploaded weights file
weights_path = '/home/piljae/kaggle/rsna-2024-lumbar-spine-degenerative-classification/resnet18-5c106cde.pth'

# Initialize models
sagittal_t1_model = CustomResNet(num_classes=3, pretrained_weights=weights_path).to(device)
axial_t2_model = CustomResNet(num_classes=3, pretrained_weights=weights_path).to(device)
sagittal_t2stir_model = CustomResNet(num_classes=3, pretrained_weights=weights_path).to(device)

# # Optionally freeze initial layers
# for param in sagittal_t1_model.model.parameters():
#     param.requires_grad = False
# for param in axial_t2_model.model.parameters():
#     param.requires_grad = False
# for param in sagittal_t2stir_model.model.parameters():
#     param.requires_grad = False

# # Unfreeze the final fully connected layer
# for param in sagittal_t1_model.model.fc.parameters():
#     param.requires_grad = True
# for param in axial_t2_model.model.fc.parameters():
#     param.requires_grad = True
# for param in sagittal_t2stir_model.model.fc.parameters():
#     param.requires_grad = True

# # Training parameters
# criterion = nn.CrossEntropyLoss()

# # Initialize separate optimizers for each model
# optimizer_sagittal_t1 = torch.optim.Adam(sagittal_t1_model.model.fc.parameters(), lr=0.001)
# optimizer_axial_t2 = torch.optim.Adam(axial_t2_model.model.fc.parameters(), lr=0.001)
# optimizer_sagittal_t2stir = torch.optim.Adam(sagittal_t2stir_model.model.fc.parameters(), lr=0.001)

# Unfreeze all layers
for param in sagittal_t1_model.model.parameters():
    param.requires_grad = True
for param in axial_t2_model.model.parameters():
    param.requires_grad = True
for param in sagittal_t2stir_model.model.parameters():
    param.requires_grad = True

# Training parameters
criterion = nn.CrossEntropyLoss()

# Initialize separate optimizers for each model with L2 regularization
optimizer_sagittal_t1 = torch.optim.SGD(sagittal_t1_model.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_axial_t2 = torch.optim.SGD(axial_t2_model.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
optimizer_sagittal_t2stir = torch.optim.SGD(sagittal_t2stir_model.model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Store the models and optimizers in dictionaries for easy access
models = {
    'Sagittal T1': sagittal_t1_model,
    'Axial T2': axial_t2_model,
    'Sagittal T2/STIR': sagittal_t2stir_model,
}

optimizers = {
    'Sagittal T1': optimizer_sagittal_t1,
    'Axial T2': optimizer_axial_t2,
    'Sagittal T2/STIR': optimizer_sagittal_t2stir,
}

trainable_params = sum(p.numel() for p in sagittal_t1_model.parameters() if p.requires_grad)
print(f"Number of parameters: {trainable_params}")



############################# Training ###################################################3

label_map = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}


# Define the number of epochs
num_epochs = 15

scheduler_sagittal_t1 = torch.optim.lr_scheduler.StepLR(optimizer_sagittal_t1, step_size=7, gamma=0.1)
scheduler_axial_t2 = torch.optim.lr_scheduler.StepLR(optimizer_axial_t2, step_size=7, gamma=0.1)
scheduler_sagittal_t2stir = torch.optim.lr_scheduler.StepLR(optimizer_sagittal_t2stir, step_size=7, gamma=0.1)

schedulers = {
    'Sagittal T1': scheduler_sagittal_t1,
    'Axial T2': scheduler_axial_t2,
    'Sagittal T2/STIR': scheduler_sagittal_t2stir,
}


# Training loop for all models
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    print("-" * 20)
    
    for model_name, model in models.items():
        
        # Set the model to training mode
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        
        trainloader = dataloaders[model_name][0]
        valloader = dataloaders[model_name][1]
        optimizer = optimizers[model_name]
        scheduler = schedulers[model_name]
        
        for images, labels in trainloader:
            labels = torch.tensor([label_map[label] for label in labels]).to(device)
            images = images.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = correct_predictions.double() / len(trainloader.dataset)
        
        print(f"{model_name} Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        scheduler.step()
        
        # Validation step
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        
        with torch.no_grad():
            for images, labels in valloader:
                labels = torch.tensor([label_map[label] for label in labels]).to(device)
                images = images.to(device)
                
                outputs = model(images)

                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct_predictions += torch.sum(preds == labels.data)
        
        val_loss = val_running_loss / len(valloader.dataset)
        val_acc = val_correct_predictions.double() / len(valloader.dataset)
        
        print(f"{model_name} Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")




##################################### Inference ##################################################################