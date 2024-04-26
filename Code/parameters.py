import torch
import torch.nn as nn
import os
from Models.Base_3DCAE import Base_3DCAE
from Models.Base_3DCAE_2 import Base_3DCAE_2
from Models.CNN_3D import CNN_3D
from Models.MultiModal_3DCAE import MultiModal_3DCAE
from Models.EarlyConcatenation_3DCAE import EarlyConcatenation_3DCAE
from Models.EarlyAddition_3DCAE import EarlyAddition_3DCAE
from Models.EarlySubtraction_3DCAE import EarlySubtraction_3DCAE
from Models.LateConcatenation_3DCAE import LateConcatenation_3DCAE
from Models.LateAddition_3DCAE import LateAddition_3DCAE
from Models.LateSubtraction_3DCAE import LateSubtraction_3DCAE

script_directory = os.path.dirname(__file__)
project_directory = os.path.dirname(script_directory)  # Base Folder - FallDetection
hdd_directory = "F:\FYP Datasets\MUVIM"  # Change to your HDD Directory

access_dataset_from_external_hdd = False
if access_dataset_from_external_hdd:
    dataset_directory = hdd_directory
else:
    dataset_directory = project_directory

ht, wd = 64, 64  # Preprocessed image dimensions

fair_comparison = True  # For using common fall and non fall folders of all modalities
metadata_set = 1  # 1 is Generated Locally, 2 is Downloaded from MUVIM Repo. Always use 1 for multi modality
TOD = "Both"  # Time of Day

frame_rate_adjusted_dataset = False  # Use dataset adjusted to 8 fps
if frame_rate_adjusted_dataset:
    dataset_category = "FPS-Adjusted"
else:
    dataset_category = "Base"

# Data augmentation
data_augmentation = False  # Enable or disable data augmentation techniques
augmentation_size = 0.5  # Ratio of folders to be augmented

# Key Frame Extraction (Always set true for CNN Models, Optional for Autoencoder Models)
key_frame_extraction = False  # Enable or disable background subtraction
key_frame_extraction_algorithms = ["BG_Subtraction", "Optical_Flow"]
key_frame_extraction_algorithm = key_frame_extraction_algorithms[0]
# Minimum Percentage of non-zero pixels required to classify as a key_frame
bg_subtraction_threshold = 0.001
# Minimum value of Optical flow required to classify as a key_frame
thermal_optical_flow_threshold = 0.00005
ip_optical_flow_threshold = 0.001
oni_ir_optical_flow_threshold = 0.00005

# Feature Extraction
background_subtraction = False  # Enable or disable background subtraction
background_subtraction_algorithms = ["GMG", "MOG2", "MOG"]
background_subtraction_algorithm = background_subtraction_algorithms[0]  # Choose the algorithm to be used

feature_extraction = background_subtraction  # Perform logical OR with future feature extraction methods' flags

batch_size = 1  # No.of video folder(s) per batch (For train and test dataloader)
window_len = 8
stride = 1
dropout = 0.25
learning_rate = 0.0002
num_epochs = 20
chunk_size = 64
forward_chunk_size = 8  # This is smaller due to memory constraints
device = "cuda" if torch.cuda.is_available() else "cpu"

anomaly_detection_model = True  # True for Autoencoder models, False for CNN models
test_size = 0.2  # Ratio of data to be taken as test data (If anomaly_detection_model is false)

models = [Base_3DCAE, Base_3DCAE_2, CNN_3D]
model = models[0]  # Choose model to be used
loss_fns = [
    nn.BCELoss(),
    nn.CrossEntropyLoss(),
    nn.MSELoss(),
    nn.L1Loss(),
    nn.HuberLoss(),
    nn.SmoothL1Loss(),
]
loss_fn = loss_fns[2]  # Choose loss function based on model used

spatial_temporal_loss = False  # Enable or disable spatial temporal loss function
# Weights used when calculating loss using spatial temporal loss function
w1 = 1
w2 = 0.00001

# Multi-modal parameters
multi_modal_models = [
    MultiModal_3DCAE,
    EarlyAddition_3DCAE,
    EarlyConcatenation_3DCAE,
    EarlySubtraction_3DCAE,
    LateAddition_3DCAE,
    LateConcatenation_3DCAE,
    LateSubtraction_3DCAE,
]
multi_modal_model = multi_modal_models[0]

if (multi_modal_models.index(multi_modal_model) > 0) & (multi_modal_models.index(multi_modal_model) < 4):
    # Early Fusion
    fusion_type = 0
else:
    # Late Fusion
    fusion_type = 1

"""
If true : For a video across both the modalities - fall frames will be adjusted to equal length and same period, 
frames before and after the fall will be adjusted to equal length and same period. 
Overall video length will be uniform. Eliminates use of pad/trim.
If false : Pad/trim flag will be used,
"""
synchronise_video = True
# If true : For a specific video across all the modalities, the video will be padded to match the modality with the maximum length video
# If false : For a specific video across all the modalities, the video will be trimmed to match the modality with the minimum length video
# Not applicable if synchronise_video is set to true
pad_video = True

"""
Preset's for quickly changing the architecture specific parameters.
0 - No preset.
1 - For Autoencoder Models
2 - For CNN Model
3 - For Multi Modal Autoencoder Model
Note - If preset > 0 , some parameters above might be overidden
"""
preset = 0
if preset == 1:
    anomaly_detection_model = True
    model = models[0]  # Index 0 or 1
    key_frame_extraction = False  # Performance is bad if True
    # Set Feature Extraction, Augmentation, Loss Function etc..
elif preset == 2:
    anomaly_detection_model = False
    model = models[2]
    loss_fn = loss_fns[0]
    key_frame_extraction = True
    # Set Feature Extraction, Augmentation, Key Frame Extraction Algorithm etc..
elif preset == 3:
    anomaly_detection_model = True
    multi_modal_model = multi_modal_models[0]  # Index 0 to 6
    if (multi_modal_models.index(multi_modal_model) > 0) & (multi_modal_models.index(multi_modal_model) < 4):
        fusion_type = 0
    else:
        fusion_type = 1
    frame_rate_adjusted_dataset = True
    dataset_category = "FPS-Adjusted"
    fair_comparison = True
    metadata_set = 1
    # Set Feature Extraction, Augmentation, Loss Function, Synchronise Video, Pad Mode etc..

"""
For Live Demo :
For single_modality_demo.ipynb - Set preset 0 with model = models[0], background_subtraction = True / False
For multi_modality_demo.ipynb - Set preset 3 with multi_modal_model = multi_modal_models[0], synchronise_video = True, background_subtraction = True / False
"""
demo = False
demo_length = 2  # No. of test videos
