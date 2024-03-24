import torch
import torch.nn as nn
import os
from Models.Base_3DCAE import Base_3DCAE
from Models.Base_3DCAE_2 import Base_3DCAE_2
from Models.CNN_3D import CNN_3D
from Models.MultiModal_3DCAE import MultiModal_3DCAE

script_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(script_directory)  # Base Folder - FallDetection
hdd_directory = "F:\FYP Datasets\MUVIM"  # Change to your HDD Directory

access_dataset_from_external_hdd = False
if access_dataset_from_external_hdd:
    dataset_directory = hdd_directory
else:
    dataset_directory = project_directory

ht, wd = 64, 64  # Preprocessed image dimensions

# Feature Extraction
background_subtraction = False  # Enable or disable background subtraction
background_subtraction_algorithms = ["GMG", "MOG2", "MOG"]
background_subtraction_algorithm = background_subtraction_algorithms[0]  # Choose the algorithm to be used

feature_extraction = background_subtraction  # Perform logical OR with future feature extraction methods' flags

# Data augmentation
data_augmentation = False  # Enable or disable data augmentation techniques

batch_size = 1  # No.of video folder(s) per batch (For train and test dataloader)

window_len = 8
stride = 1
fair_comparison = True  # For using common fall and non fall folders of all modalities
metadata_set = 2  # 1 is Generated Locally, 2 is Downloaded from MUVIM Repo
TOD = "Both"  # Time of Day

device = "cuda" if torch.cuda.is_available() else "cpu"

anomaly_detection_model = False  # True for Autoencoder models, False for CNN models
test_size = 0.2  # Ratio of data to be taken as test data (If anomaly_detection_model is false)
key_frame_threshold = 0.001  # Percentage of non-zero pixels required to classify as key_frame

models = [Base_3DCAE, Base_3DCAE_2, CNN_3D]
model = models[2]  # Choose model to be used

dropout = 0.25
learning_rate = 0.0002
num_epochs = 20
chunk_size = 64
forward_chunk_size = 8  # This is smaller due to memory constraints

loss_fns = [
    nn.BCELoss(),
    nn.CrossEntropyLoss(),
    nn.MSELoss(),
    nn.L1Loss(),
    nn.HuberLoss(),
    nn.SmoothL1Loss(),
]
loss_fn = loss_fns[0]  # Choose loss function based on model used

spatial_temporal_loss = False  # Enable or disable spatial temporal loss function
# Weights used when calculating loss using spatial temporal loss function
w1 = 1
w2 = 0.00001

# Multi-modal parameters
multi_modal_models = [MultiModal_3DCAE]
multi_modal_model = multi_modal_models[0]
frame_rate_adjusted_dataset = False  # Use dataset adjusted to 8 fps
# If true : For a specific video across all the modalities, the video will be padded to match the modality with the maximum length video
# If false : For a specific video across all the modalities, the video will be trimmed to match the modality with the minimum length video
pad_video = False
