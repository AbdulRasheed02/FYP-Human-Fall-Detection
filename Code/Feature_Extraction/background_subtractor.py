import cv2
import h5py
import os
import sys
import numpy as np

# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from parameters import (
    project_directory,
    dataset_directory,
    background_subtraction_algorithms,
    background_subtraction_algorithm,
    dataset_category,
)

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


# https://docs.opencv.org/4.x/d1/d5c/classcv_1_1bgsegm_1_1BackgroundSubtractorGMG.html
def perform_background_subtraction_GMG(vid_total):
    background_subtracted_vid_total = []

    # Create background subtractor
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # bg_subtractor.setDecisionThreshold(0.95)  # Default - 0.8
    # bg_subtractor.setDefaultLearningRate(0.025)  # Default - 0.025
    # bg_subtractor.setNumFrames(100)  # Default - 120

    for frame in vid_total:
        frame = np.array(frame, dtype=np.float32)
        # print(frame.shape)
        # Perform background subtraction
        foreground_mask = bg_subtractor.apply(frame)
        foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
        background_subtracted_vid_total.append(foreground_mask)

        # # To view the images
        # cv2.imshow("Original Frame", frame)
        # cv2.imshow("Foreground Mask - GMG", foreground_mask)
        # # Exit on 'q' press
        # k = cv2.waitKey(30) & 0xFF
        # if k == 27:
        #     break

    return background_subtracted_vid_total


# https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html
def perform_background_subtraction_MOG2(vid_total):
    background_subtracted_vid_total = []

    # Create background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    # Sets the number of last frames that affect the background model.
    bg_subtractor.setHistory(history=50)  # Default - 500
    bg_subtractor.setDetectShadows(False)  # Default - True
    # Sets the variance threshold for the pixel-model match. (Pixels whose variance is above this value will be in foreground)
    bg_subtractor.setVarThreshold(200)  # Default - 16.0

    for frame in vid_total:
        # Generated foreground is always black due to the preprocessing step - (img = img.astype("float32") / 255 ). So we multiply by 255 to get better foreground mask.
        frame = frame * 255
        # print(frame.shape)
        # Perform background subtraction. learningRate = -1 is default
        foreground_mask = bg_subtractor.apply(frame, learningRate=-1)
        background_subtracted_vid_total.append(foreground_mask)

        # # To view the images
        # cv2.imshow("Original Frame", frame)
        # cv2.imshow("Foreground Mask - MOG2", foreground_mask)
        # # Exit on 'q' press
        # k = cv2.waitKey(30) & 0xFF
        # if k == 27:
        #     break

    return background_subtracted_vid_total


# https://docs.opencv.org/4.x/d6/da7/classcv_1_1bgsegm_1_1BackgroundSubtractorMOG.html
def perform_background_subtraction_MOG(vid_total):
    background_subtracted_vid_total = []

    # Create background subtractor
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    # Sets the number of last frames that affect the background model.
    # bg_subtractor.setHistory(100)  # Default - 200
    # bg_subtractor.setNMixtures(5) # Default - 5
    # bg_subtractor.setBackgroundRatio(0.95)  # Default - 0.7

    for frame in vid_total:
        # Generated foreground is always black due to the preprocessing step - (img = img.astype("float32") / 255 ). So we multiply by 255 to get better foreground mask.
        frame = frame * 255
        # uint8 is the required input type for MOG
        frame = np.array(frame, dtype=np.uint8)
        # print(frame.shape)
        # Perform background subtraction.
        foreground_mask = bg_subtractor.apply(frame)
        background_subtracted_vid_total.append(foreground_mask)

        # # To view the images
        # cv2.imshow("Original Frame", frame)
        # cv2.imshow("Foreground Mask - MOG", foreground_mask)
        # # Exit on 'q' press
        # k = cv2.waitKey(30) & 0xFF
        # if k == 27:
        #     break

    return background_subtracted_vid_total


def perform_background_subtraction(vid_total):
    if background_subtraction_algorithm == background_subtraction_algorithms[0]:
        return perform_background_subtraction_GMG(vid_total)
    if background_subtraction_algorithm == background_subtraction_algorithms[1]:
        return perform_background_subtraction_MOG2(vid_total)
    if background_subtraction_algorithm == background_subtraction_algorithms[2]:
        return perform_background_subtraction_MOG(vid_total)


# Uncomment the lines below to run this file independently (For development and debugging)

# # For using preprocessed images from h5py as input
# name = "Thermal_T3"
# path = "{}\Dataset\H5PY\{}_Data_set-{}-imgdim64x64.h5".format(project_directory, dataset_category, name)
# with h5py.File(path, "r") as hf:
#     data_dict = hf["{}/Processed/Split_by_video".format(name)]
#     # Any fall or ADL directory
#     vid_total = data_dict["Fall0"]["Data"][:]
#     background_subtracted_vid_total = perform_background_subtraction(vid_total)

# # For using original images from the dataset as input
# modality = "Thermal"
# # Any fall or ADL directory
# folder_location = "{}\Dataset\Fall-Data\{}\{}\Fall\Fall0".format(dataset_directory, dataset_category, modality)
# vid_total = []
# for filename in os.listdir(folder_location):
#     # Get the full path of the image
#     img_path = os.path.join(folder_location, filename)
#     # Read the image using cv2.imread and in greyscale
#     img = cv2.imread(img_path, 0)
#     # Check if image loaded successfully (cv2.imread returns None on error)
#     if img is not None:
#         vid_total.append(img)
# background_subtracted_vid_total = perform_background_subtraction(vid_total)
