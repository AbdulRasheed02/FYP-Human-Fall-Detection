import torch
import torch.nn as nn
import os
from Models.Base_3DCAE import Base_3DCAE
from Models.Base_3DCAE_2 import Base_3DCAE_2
from Models.CNN_3D import CNN_3D

script_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(script_directory)  # Base Folder - FallDetection
hdd_directory = "F:\FYP Datasets\MUVIM"  # Change to your HDD Directory

access_dataset_from_external_hdd = False
if access_dataset_from_external_hdd:
    dataset_directory = hdd_directory
else:
    dataset_directory = project_directory

ht, wd = 64, 64  # Preprocessed image dimensions

anomaly_detection_model = False  # True for Autoencoder models, False for CNN models
test_size = 0.2  # Ratio of data to be taken as test data (If anomaly_detection_model is false)

# Feature Extraction
feature_extraction = False  # Enable or disable feature extraction techniques
background_subtraction = False  # Enable or disable background subtraction
background_subtraction_algorithms = ["GMG", "MOG2", "MOG"]
background_subtraction_algorithm = background_subtraction_algorithms[0]  # Choose the algorithm to be used

# Data augmentation
data_augmentation = False  # Enable or disable data augmentation techniques

batch_size = 1  # No.of video folder(s) per batch (For train and test dataloader)

window_len = 8
stride = 1
fair_comparison = True  # For using common fall and non fall folders of all modalities
metadata_set = 2  # 1 is Generated Locally, 2 is Downloaded from MUVIM Repo
TOD = "Both"  # Time of Day

device = "cuda" if torch.cuda.is_available() else "cpu"

models = [Base_3DCAE, Base_3DCAE_2, CNN_3D]
model = models[2]  # Choose model to be used

dropout = 0.25
learning_rate = 0.0002
num_epochs = 20
chunk_size = 64
forward_chunk = 8
forward_chunk_size = 8  # This is smaller due to memory constraints

loss_fns = [
    nn.MSELoss(),
    nn.L1Loss(),
    nn.BCELoss(),
    nn.HuberLoss(),
    nn.SmoothL1Loss(),
]
loss_fn = loss_fns[1]  # Choose loss function based on model used

spatial_temporal_loss = False  # Enable or disable spatial temporal loss function
# Weights used when calculating loss using spatial temporal loss function
w1 = 1
w2 = 0.00001
