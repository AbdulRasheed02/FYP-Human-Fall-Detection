import torch
import torch.nn as nn
import os

script_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(script_directory)  # Base Folder - FallDetection
hdd_directory = "F:\FYP Datasets\MUVIM"  # Change to your HDD Directory

access_dataset_from_external_hdd = False
if access_dataset_from_external_hdd:
    dataset_directory = hdd_directory
else:
    dataset_directory = project_directory

ht, wd = 64, 64  # Preprocessed image dimensions

batch_size = 1  # No.of samples per batch (For train and test dataloader)

window_len = 8
stride = 1
fair_comparison = True
TOD = "Both"  # Time of Day

device = "cuda" if torch.cuda.is_available() else "cpu"

dropout = 0.25
learning_rate = 0.0002
num_epochs = 20
chunk_size = 64
forward_chunk = 8
forward_chunk_size = 8  # This is smaller due to memory constrains

loss_fn = nn.MSELoss()
