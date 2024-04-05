import cv2
import numpy as np
import sys
import os
import h5py
import re
from glob import glob

# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from parameters import (
    project_directory,
    dataset_directory,
    dataset_category,
)

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def calculate_optical_flow(vid_total):
    optical_flow_values = []
    # Optical flow cannot be calculated for first frame
    optical_flow_values.append(0)
    for index in range(1, len(vid_total)):
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(vid_total[index - 1], vid_total[index], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sum(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2))
        optical_flow_values.append(magnitude)
    return optical_flow_values


# Uncomment the lines below to run this file independently (For development and debugging)

# # For using preprocessed images from h5py as input
# name = "Thermal_T3"
# path = "{}\Dataset\H5PY\{}_Data_set-{}-imgdim64x64.h5".format(project_directory, dataset_category, name)
# with h5py.File(path, "r") as hf:
#     data_dict = hf["{}/Processed/Split_by_video".format(name)]
#     # Any fall or ADL directory
#     vid_total = data_dict["Fall0"]["Data"][:]
#     optical_flow_values = calculate_optical_flow(vid_total)
