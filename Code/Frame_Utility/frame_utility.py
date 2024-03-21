import numpy as np
import cv2
import h5py
import sys
import os

# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from parameters import project_directory, key_frame_threshold
from Feature_Extraction.background_subtractor import perform_background_subtraction

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def fall_frame_extractor(vid_total, labels_total):

    # Get indices where labels_total == 1. [0] is indices where condition is true
    fall_indices = np.where(labels_total == 1)[0]
    vid_total = vid_total[fall_indices]  # Filter vid_total using the indices
    labels_total = labels_total[fall_indices]  # Filter labels_total using the indices

    vid_total = vid_total.tolist()
    labels_total = labels_total.tolist()

    # Will throw error if length of vid_total is less than 10
    if len(vid_total) < 10:
        last_fall_frame = vid_total[-1]
        # Append last_fall_frame until length reaches 10
        while len(vid_total) < 10:
            vid_total.append(last_fall_frame)
            labels_total.append(1)

    vid_total = np.array(vid_total)
    labels_total = np.array(labels_total)

    # # View fall frames
    # for fall_frame in vid_total:
    #     cv2.imshow("Fall frame", fall_frame)
    #     # Exit on 'q' press
    #     k = cv2.waitKey(30) & 0xFF
    #     if k == 27:
    #         break

    return vid_total, labels_total


def key_frame_extractor(vid_total, labels_total, threshold):
    background_subtracted_vid_total = perform_background_subtraction(vid_total)
    key_frames = []
    key_frame_indices = []

    for index, frame in enumerate(background_subtracted_vid_total):
        # Calculate movement ratio (percentage of non-zero pixels)
        movement_ratio = np.count_nonzero(frame) / (frame.shape[0] * frame.shape[1])
        if movement_ratio > threshold:  # Extract keyframe based on movement threshold
            key_frames.append(frame)
            key_frame_indices.append(index)
            # cv2.imshow("Key frame", frame)
            # # Exit on 'q' press
            # k = cv2.waitKey(30) & 0xFF
            # if k == 27:
            #     break

    # Will throw error if length of vid_total is less than 10
    if len(key_frame_indices) < 10:
        # If key_frame_indices has any element, repeat that index. Else pad it with zeros
        last_element = key_frame_indices[-1] if key_frame_indices else 0
        key_frame_indices = key_frame_indices + [last_element] * (10 - len(key_frame_indices))

    # Extract original keyframes from vid_total using key_frame_indices
    vid_total = [vid_total[i] for i in key_frame_indices]
    # Extract corresponding labels from labels_total using key_frame_indices
    labels_total = [labels_total[i] for i in key_frame_indices]

    # # Can directly use background subtracted key_frames
    # vid_total = key_frames
    # labels_total = [labels_total[i] for i in key_frame_indices]

    return vid_total, labels_total


# # For using preprocessed images from h5py as input
# name = "Thermal_T3"
# path = "{}\Dataset\H5PY\Data_set-{}-imgdim64x64.h5".format(project_directory, name)
# with h5py.File(path, "r") as hf:
#     data_dict = hf["{}/Processed/Split_by_video".format(name)]
#     # Any Fall directory
#     # vid_total_fall = data_dict["Fall0"]["Data"][:]
#     # labels_total_fall = data_dict["Fall0"]["Labels"][:]
#     # fall_frame_extractor(vid_total_fall, labels_total_fall)
#     # # Any ADL directory
#     # vid_total_non_fall = data_dict["NonFall0"]["Data"][:]
#     # labels_total_non_fall = data_dict["NonFall0"]["Labels"][:]
#     # key_frame_extractor(vid_total_non_fall, labels_total_non_fall, key_frame_threshold)
