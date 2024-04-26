import numpy as np
import cv2
import h5py
import sys
import os

# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from parameters import (
    project_directory,
    bg_subtraction_threshold,
    dataset_category,
    key_frame_extraction_algorithms,
    key_frame_extraction_algorithm,
    thermal_optical_flow_threshold,
    oni_ir_optical_flow_threshold,
    ip_optical_flow_threshold,
)
from Feature_Extraction.background_subtractor import perform_background_subtraction
from Feature_Extraction.optical_flow import calculate_optical_flow

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def fall_frame_extractor(vid_total, labels_total):

    # Get indices where labels_total == 1. [0] is indices where condition is true
    fall_indices = (np.where(labels_total == 1)[0]).tolist()

    # To prevent downstream error when length of vid_total is less than 10
    if len(fall_indices) < 10:
        # If fall_indices has any element, choose the last element. Else select 0
        pad_element_index = fall_indices[-1] if fall_indices else 0
        # Pad until length is 10
        fall_indices = fall_indices + [pad_element_index] * (10 - len(fall_indices))

    vid_total = vid_total[fall_indices]  # Filter vid_total using the indices
    labels_total = labels_total[fall_indices]  # Filter labels_total using the indices

    # print(len(fall_indices), len(vid_total), len(labels_total))

    # # View fall frames
    # for fall_frame in vid_total:
    #     cv2.imshow("Fall frame", fall_frame)
    #     # Exit on 'q' press
    #     k = cv2.waitKey(30) & 0xFF
    #     if k == 27:
    #         break

    return vid_total, labels_total


def key_frame_extractor(vid_total, labels_total, modality):
    key_frame_indices = []

    if key_frame_extraction_algorithm == key_frame_extraction_algorithms[0]:
        background_subtracted_vid_total = perform_background_subtraction(vid_total)

        for index, frame in enumerate(background_subtracted_vid_total):
            # Calculate movement ratio (percentage of non-zero pixels)
            movement_ratio = np.count_nonzero(frame) / (frame.shape[0] * frame.shape[1])
            if movement_ratio > bg_subtraction_threshold:  # Extract keyframe based on movement threshold
                key_frame_indices.append(index)
                # cv2.imshow("Key frame", frame)
                # # Exit on 'q' press
                # k = cv2.waitKey(30) & 0xFF
                # if k == 27:
                #     break

    elif key_frame_extraction_algorithm == key_frame_extraction_algorithms[1]:
        optical_flow_values = calculate_optical_flow(vid_total)

        if modality == "Thermal_T3":
            optical_flow_threshold = thermal_optical_flow_threshold
        elif modality == "ONI_IR_T":
            optical_flow_threshold = oni_ir_optical_flow_threshold
        elif modality == "IP_T":
            optical_flow_threshold = ip_optical_flow_threshold
        # print(optical_flow_threshold)

        for index, magnitude in enumerate(optical_flow_values):
            if magnitude > optical_flow_threshold:
                key_frame_indices.append(index)
                # cv2.imshow("Key frame", vid_total[index])
                # # Exit on 'q' press
                # k = cv2.waitKey(30) & 0xFF
                # if k == 27:
                #     break

    # To prevent downstream error when length of vid_total is less than 10
    if len(key_frame_indices) < 10:
        # If key_frame_indices has any element, choose the last element. Else select 0
        pad_element_index = key_frame_indices[-1] if key_frame_indices else 0
        # Pad until length is 10
        key_frame_indices = key_frame_indices + [pad_element_index] * (10 - len(key_frame_indices))

    # Extract original keyframes from vid_total using key_frame_indices
    vid_total = [vid_total[i] for i in key_frame_indices]
    # Extract corresponding labels from labels_total using key_frame_indices
    labels_total = [labels_total[i] for i in key_frame_indices]

    if key_frame_extraction_algorithm == key_frame_extraction_algorithms[0]:
        # If key frame extraction algorithm was background subtraction, Extract background subtracted keyframes using key_frame_indices
        background_subtracted_key_frames = [background_subtracted_vid_total[i] for i in key_frame_indices]
    else:
        # If key frame extraction algorithm was not background subtraction, return empty list. BUT DO NOT MAKE USE OF IT
        background_subtracted_key_frames = []

    return vid_total, np.array(background_subtracted_key_frames), labels_total


# Helper Function - Divides interval between first_index and last_index into 'count' number of equally spaced indices
# Ex - First frame - 0, Last frame - 20, Count - 5. modified_indices - [0,5,10,15,20]
def equalise_frames_helper(first_index, last_index, vid_total, labels_total, count):
    modified_indices = np.linspace(first_index, last_index, num=count)
    modified_indices = np.round(modified_indices).astype(int)
    modified_frames = [vid_total[i] for i in modified_indices]
    modified_frame_labels = [labels_total[i] for i in modified_indices]
    return modified_frames, modified_frame_labels, modified_indices


"""
Helper Function - To equalise videos which contains two falls. If a video contain two falls, 
structure of video is - (adl->fall->adl->fall->adl).
First fall is equalised, then frames between the two falls is equalised, then the second fall is equalised. 
"""


def equalise_two_falls_helper(
    vid_total_modality_1,
    vid_total_modality_2,
    labels_total_modality_1,
    labels_total_modality_2,
    diff_modality_1,
    diff_modality_2,
):
    # Compute necessary frame indices and lengths
    fall_1_first_frame_modality_1 = (np.where(diff_modality_1 == 1)[0][0]) + 1
    fall_1_last_frame_modality_1 = np.where(diff_modality_1 == -1)[0][0]
    fall_2_first_frame_modality_1 = (np.where(diff_modality_1 == 1)[0][1]) + 1
    fall_2_last_frame_modality_1 = np.where(diff_modality_1 == -1)[0][1]

    fall_1_first_frame_modality_2 = (np.where(diff_modality_2 == 1)[0][0]) + 1
    fall_1_last_frame_modality_2 = np.where(diff_modality_2 == -1)[0][0]
    fall_2_first_frame_modality_2 = (np.where(diff_modality_2 == 1)[0][1]) + 1
    fall_2_last_frame_modality_2 = np.where(diff_modality_2 == -1)[0][1]

    fall_1_length_modality_1 = fall_1_last_frame_modality_1 - fall_1_first_frame_modality_1 + 1
    fall_2_length_modality_1 = fall_2_last_frame_modality_1 - fall_2_first_frame_modality_1 + 1
    fall_1_length_modality_2 = fall_1_last_frame_modality_2 - fall_1_first_frame_modality_2 + 1
    fall_2_length_modality_2 = fall_2_last_frame_modality_2 - fall_2_first_frame_modality_2 + 1

    intermediate_length_modality_1 = fall_2_first_frame_modality_1 - fall_1_last_frame_modality_1 - 1
    intermediate_length_modality_2 = fall_2_first_frame_modality_2 - fall_1_last_frame_modality_2 - 1

    ## Print Statements

    # print("Double Fall\nFall 1 : Start -", fall_1_first_frame_modality_1,"End -", fall_1_last_frame_modality_1,", Start -", fall_1_first_frame_modality_2,"End -", fall_1_last_frame_modality_2)  # fmt: skip
    # print("Fall 2 : Start -", fall_2_first_frame_modality_1,"End -", fall_2_last_frame_modality_1,", Start -", fall_2_first_frame_modality_2,"End -", fall_2_last_frame_modality_2)  # fmt: skip
    # print("Fall 1 Length -", fall_1_length_modality_1, fall_1_length_modality_2)
    # print("Intermediate Length -", intermediate_length_modality_1, intermediate_length_modality_2)
    # print("Fall 2 Length -", fall_2_length_modality_1, fall_2_length_modality_2)

    ## Four Steps

    # 1. Equalise frames of first fall
    if fall_1_length_modality_1 >= fall_1_length_modality_2:
        # Modality 1 has longer length, so decrease it to match Modality 2 length
        fall_1_frames_modality_1, fall_1_frame_labels_modality_1, fall_1_frame_indices_modality_1 = (
            equalise_frames_helper(
                fall_1_first_frame_modality_1,
                fall_1_last_frame_modality_1,
                vid_total_modality_1,
                labels_total_modality_1,
                count=fall_1_length_modality_2,
            )
        )
        # Extract unmodified frames of first fall for Modality 2
        fall_1_frames_modality_2 = vid_total_modality_2[
            fall_1_first_frame_modality_2 : fall_1_last_frame_modality_2 + 1
        ]
        fall_1_frame_labels_modality_2 = labels_total_modality_2[
            fall_1_first_frame_modality_2 : fall_1_last_frame_modality_2 + 1
        ]
        fall_1_frame_indices_modality_2 = np.arange(fall_1_first_frame_modality_2, fall_1_last_frame_modality_2 + 1)
    else:
        # Modality 2 has longer length, so decrease it to match Modality 1 length
        fall_1_frames_modality_2, fall_1_frame_labels_modality_2, fall_1_frame_indices_modality_2 = (
            equalise_frames_helper(
                fall_1_first_frame_modality_2,
                fall_1_last_frame_modality_2,
                vid_total_modality_2,
                labels_total_modality_2,
                count=fall_1_length_modality_1,
            )
        )
        # Extract unmodified frames of first fall for Modality 1
        fall_1_frames_modality_1 = vid_total_modality_1[
            fall_1_first_frame_modality_1 : fall_1_last_frame_modality_1 + 1
        ]
        fall_1_frame_labels_modality_1 = labels_total_modality_1[
            fall_1_first_frame_modality_1 : fall_1_last_frame_modality_1 + 1
        ]
        fall_1_frame_indices_modality_1 = np.arange(fall_1_first_frame_modality_1, fall_1_last_frame_modality_1 + 1)

    # 2. Equalise intermediate frames (Same logic as Step 1)
    if intermediate_length_modality_1 >= intermediate_length_modality_2:
        intermediate_frames_modality_1, intermediate_frame_labels_modality_1, intermediate_frame_indices_modality_1 = (
            equalise_frames_helper(
                fall_1_last_frame_modality_1 + 1,
                fall_2_first_frame_modality_1 - 1,
                vid_total_modality_1,
                labels_total_modality_1,
                count=intermediate_length_modality_2,
            )
        )
        intermediate_frames_modality_2 = vid_total_modality_2[
            fall_1_last_frame_modality_2 + 1 : fall_2_first_frame_modality_2
        ]
        intermediate_frame_labels_modality_2 = labels_total_modality_2[
            fall_1_last_frame_modality_2 + 1 : fall_2_first_frame_modality_2
        ]
        intermediate_frame_indices_modality_2 = np.arange(
            fall_1_last_frame_modality_2 + 1, fall_2_first_frame_modality_2
        )
    else:
        intermediate_frames_modality_2, intermediate_frame_labels_modality_2, intermediate_frame_indices_modality_2 = (
            equalise_frames_helper(
                fall_1_last_frame_modality_2 + 1,
                fall_2_first_frame_modality_2 - 1,
                vid_total_modality_2,
                labels_total_modality_2,
                count=intermediate_length_modality_1,
            )
        )
        intermediate_frames_modality_1 = vid_total_modality_1[
            fall_1_last_frame_modality_1 + 1 : fall_2_first_frame_modality_1
        ]
        intermediate_frame_labels_modality_1 = labels_total_modality_1[
            fall_1_last_frame_modality_1 + 1 : fall_2_first_frame_modality_1
        ]
        intermediate_frame_indices_modality_1 = np.arange(
            fall_1_last_frame_modality_1 + 1, fall_2_first_frame_modality_1
        )

    # 3. Equalise frames of second fall (Same logic as Step 1)
    if fall_2_length_modality_1 >= fall_2_length_modality_2:
        fall_2_frames_modality_1, fall_2_frame_labels_modality_1, fall_2_frame_indices_modality_1 = (
            equalise_frames_helper(
                fall_2_first_frame_modality_1,
                fall_2_last_frame_modality_1,
                vid_total_modality_1,
                labels_total_modality_1,
                count=fall_2_length_modality_2,
            )
        )
        fall_2_frames_modality_2 = vid_total_modality_2[
            fall_2_first_frame_modality_2 : fall_2_last_frame_modality_2 + 1
        ]
        fall_2_frame_labels_modality_2 = labels_total_modality_2[
            fall_2_first_frame_modality_2 : fall_2_last_frame_modality_2 + 1
        ]
        fall_2_frame_indices_modality_2 = np.arange(fall_2_first_frame_modality_2, fall_2_last_frame_modality_2 + 1)
    else:
        fall_2_frames_modality_2, fall_2_frame_labels_modality_2, fall_2_frame_indices_modality_2 = (
            equalise_frames_helper(
                fall_2_first_frame_modality_2,
                fall_2_last_frame_modality_2,
                vid_total_modality_2,
                labels_total_modality_2,
                count=fall_2_length_modality_1,
            )
        )
        fall_2_frames_modality_1 = vid_total_modality_1[
            fall_2_first_frame_modality_1 : fall_2_last_frame_modality_1 + 1
        ]
        fall_2_frame_labels_modality_1 = labels_total_modality_1[
            fall_2_first_frame_modality_1 : fall_2_last_frame_modality_1 + 1
        ]
        fall_2_frame_indices_modality_1 = np.arange(fall_2_first_frame_modality_1, fall_2_last_frame_modality_1 + 1)

    # 4. Concatenate the three slices (for both modalities) and return as a single array (Which is returned for the Step 2 of the Main Function)

    fall_frames_modality_1 = np.concatenate(
        (
            fall_1_frames_modality_1,
            intermediate_frames_modality_1,
            fall_2_frames_modality_1,
        ),
        axis=0,
    )
    fall_frame_labels_modality_1 = np.concatenate(
        (
            fall_1_frame_labels_modality_1,
            intermediate_frame_labels_modality_1,
            fall_2_frame_labels_modality_1,
        ),
        axis=0,
    )
    fall_frame_indices_modality_1 = np.concatenate(
        (
            fall_1_frame_indices_modality_1,
            intermediate_frame_indices_modality_1,
            fall_2_frame_indices_modality_1,
        ),
        axis=0,
    )

    fall_frames_modality_2 = np.concatenate(
        (
            fall_1_frames_modality_2,
            intermediate_frames_modality_2,
            fall_2_frames_modality_2,
        ),
        axis=0,
    )
    fall_frame_labels_modality_2 = np.concatenate(
        (
            fall_1_frame_labels_modality_2,
            intermediate_frame_labels_modality_2,
            fall_2_frame_labels_modality_2,
        ),
        axis=0,
    )
    fall_frame_indices_modality_2 = np.concatenate(
        (
            fall_1_frame_indices_modality_2,
            intermediate_frame_indices_modality_2,
            fall_2_frame_indices_modality_2,
        ),
        axis=0,
    )

    return (
        fall_frames_modality_1,
        fall_frame_labels_modality_1,
        fall_frames_modality_2,
        fall_frame_labels_modality_2,
        fall_frame_indices_modality_1,
        fall_frame_indices_modality_2,
    )


def sync_frames(vid_total_modality_1, vid_total_modality_2, labels_total_modality_1, labels_total_modality_2):

    ### ADL Video (Class label 1 is not present in labels_total array)
    # Condition is satisifed if either one of the labels_total array has no class label 1
    if not ((np.any(labels_total_modality_1 == 1)) & (np.any(labels_total_modality_2 == 1))):
        # print("Original Length - ", len(vid_total_modality_1), len(vid_total_modality_2))

        ## Edge Case - One modality has one/two fall(s), other modality has no fall.
        # In such case, truncate video from the first fall for the modality with one/two fall(s). Essentially, video will have no fall now.
        # Modality 1 has one/two fall(s)
        if np.any(labels_total_modality_1 == 1):
            first_fall_frame_modality_1 = (np.where(labels_total_modality_1 == 1)[0]).tolist()[0]
            vid_total_modality_1 = vid_total_modality_1[:first_fall_frame_modality_1]
            labels_total_modality_1 = labels_total_modality_1[:first_fall_frame_modality_1]
        # Modality 2 has one/two fall(s)
        elif np.any(labels_total_modality_2 == 1):
            first_fall_frame_modality_2 = (np.where(labels_total_modality_2 == 1)[0]).tolist()[0]
            vid_total_modality_2 = vid_total_modality_2[:first_fall_frame_modality_2]
            labels_total_modality_2 = labels_total_modality_2[:first_fall_frame_modality_2]

        video_length_modality_1 = len(vid_total_modality_1)
        video_length_modality_2 = len(vid_total_modality_2)

        if video_length_modality_1 >= video_length_modality_2:
            vid_total_modality_1, labels_total_modality_1, original_indices_modality_1 = equalise_frames_helper(
                0, video_length_modality_1 - 1, vid_total_modality_1, labels_total_modality_1, video_length_modality_2
            )
            original_indices_modality_2 = np.arange(0, video_length_modality_2)
        else:
            vid_total_modality_2, labels_total_modality_2, original_indices_modality_2 = equalise_frames_helper(
                0, video_length_modality_2 - 1, vid_total_modality_2, labels_total_modality_2, video_length_modality_1
            )
            original_indices_modality_1 = np.arange(0, video_length_modality_1)

        # print("*Modified Length - ", len(vid_total_modality_1), len(vid_total_modality_2))
        return (
            vid_total_modality_1,
            vid_total_modality_2,
            labels_total_modality_1,
            labels_total_modality_2,
            original_indices_modality_1,
            original_indices_modality_2,
        )

    ### Fall Video

    ## Check if video contains 1 fall or 2 falls
    # Calculate difference between consecutive elements (b[i]=a[i+1]-a[i])
    diff_modality_1 = np.diff(labels_total_modality_1)
    diff_modality_2 = np.diff(labels_total_modality_2)
    # Find transitions by using number of non zero elements.
    # One fall structure : 0->1->0 (Diff will have two non zero elements / transitions)
    # Two falls structure : 0->1->0->1->0 (Diff will have four non zero elements / transitions)
    transitions_modality_1 = np.count_nonzero(diff_modality_1)
    transitions_modality_2 = np.count_nonzero(diff_modality_2)

    ## Edge Case - One modality has one fall, other modality has two fall.
    # In such case, truncate video from the second fall for the modality with two falls. Essentially, video will only have one fall now.
    if transitions_modality_1 != transitions_modality_2:
        # Modality 1 has two falls
        if transitions_modality_1 == 4:
            fall_2_first_frame_modality_1 = (np.where(diff_modality_1 == 1)[0][1]) + 1
            # Truncate video from second fall
            vid_total_modality_1 = vid_total_modality_1[:fall_2_first_frame_modality_1]
            labels_total_modality_1 = labels_total_modality_1[:fall_2_first_frame_modality_1]
        # Modality 2 has two falls
        else:
            fall_2_first_frame_modality_2 = (np.where(diff_modality_2 == 1)[0][1]) + 1
            # Truncate video from second fall
            vid_total_modality_2 = vid_total_modality_2[:fall_2_first_frame_modality_2]
            labels_total_modality_2 = labels_total_modality_2[:fall_2_first_frame_modality_2]

        transitions_modality_1 = transitions_modality_2 = 2

    # Compute necessary frame indices and lengths
    first_fall_frame_modality_1 = (np.where(labels_total_modality_1 == 1)[0]).tolist()[0]
    last_fall_frame_modality_1 = (np.where(labels_total_modality_1 == 1)[0]).tolist()[-1]
    first_fall_frame_modality_2 = (np.where(labels_total_modality_2 == 1)[0]).tolist()[0]
    last_fall_frame_modality_2 = (np.where(labels_total_modality_2 == 1)[0]).tolist()[-1]

    frames_before_fall_modality_1 = first_fall_frame_modality_1 - 1
    frames_before_fall_modality_2 = first_fall_frame_modality_2 - 1
    frames_after_fall_modality_1 = (len(vid_total_modality_1) - 1) - last_fall_frame_modality_1
    frames_after_fall_modality_2 = (len(vid_total_modality_2) - 1) - last_fall_frame_modality_2

    fall_length_modality_1 = (labels_total_modality_1 == 1).sum()
    fall_length_modality_2 = (labels_total_modality_2 == 1).sum()

    ## Print Statements

    # print("Original Length - ", len(vid_total_modality_1), len(vid_total_modality_2))
    # print("Frames before Fall -", frames_before_fall_modality_1 + 1, frames_before_fall_modality_2 + 1)
    # print("Fall Frames - ", fall_length_modality_1, fall_length_modality_2)
    # print("Frames after Fall -", frames_after_fall_modality_1, frames_after_fall_modality_2)

    ## Four Steps

    # 1. Equalise frames before the fall event across both modalities

    if frames_before_fall_modality_1 >= frames_before_fall_modality_2:
        before_fall_frames_modality_1, before_fall_frame_labels_modality_1, before_fall_frame_indices_modality_1 = (
            equalise_frames_helper(
                0,
                frames_before_fall_modality_1,
                vid_total_modality_1,
                labels_total_modality_1,
                count=frames_before_fall_modality_2 + 1,
            )
        )
        before_fall_frames_modality_2 = vid_total_modality_2[: frames_before_fall_modality_2 + 1]
        before_fall_frame_labels_modality_2 = labels_total_modality_2[: frames_before_fall_modality_2 + 1]
        before_fall_frame_indices_modality_2 = np.arange(0, frames_before_fall_modality_2 + 1)
    else:
        before_fall_frames_modality_2, before_fall_frame_labels_modality_2, before_fall_frame_indices_modality_2 = (
            equalise_frames_helper(
                0,
                frames_before_fall_modality_2,
                vid_total_modality_2,
                labels_total_modality_2,
                count=frames_before_fall_modality_1 + 1,
            )
        )
        before_fall_frames_modality_1 = vid_total_modality_1[: frames_before_fall_modality_1 + 1]
        before_fall_frame_labels_modality_1 = labels_total_modality_1[: frames_before_fall_modality_1 + 1]
        before_fall_frame_indices_modality_1 = np.arange(0, frames_before_fall_modality_1 + 1)

    # 2. Equalise the fall frames across both modalities

    # Video contains one fall
    if transitions_modality_1 == transitions_modality_2 == 2:
        # print("Single Fall : Start -", first_fall_frame_modality_1,"End -", last_fall_frame_modality_1,", Start -", first_fall_frame_modality_2,"End -", last_fall_frame_modality_2)  # fmt: skip

        if fall_length_modality_1 >= fall_length_modality_2:
            fall_frames_modality_1, fall_frame_labels_modality_1, fall_frame_indices_modality_1 = (
                equalise_frames_helper(
                    first_fall_frame_modality_1,
                    last_fall_frame_modality_1,
                    vid_total_modality_1,
                    labels_total_modality_1,
                    count=fall_length_modality_2,
                )
            )
            fall_frames_modality_2 = vid_total_modality_2[first_fall_frame_modality_2 : last_fall_frame_modality_2 + 1]
            fall_frame_labels_modality_2 = labels_total_modality_2[
                first_fall_frame_modality_2 : last_fall_frame_modality_2 + 1
            ]
            fall_frame_indices_modality_2 = np.arange(first_fall_frame_modality_2, last_fall_frame_modality_2 + 1)
        else:
            fall_frames_modality_2, fall_frame_labels_modality_2, fall_frame_indices_modality_2 = (
                equalise_frames_helper(
                    first_fall_frame_modality_2,
                    last_fall_frame_modality_2,
                    vid_total_modality_2,
                    labels_total_modality_2,
                    count=fall_length_modality_1,
                )
            )
            fall_frames_modality_1 = vid_total_modality_1[first_fall_frame_modality_1 : last_fall_frame_modality_1 + 1]
            fall_frame_labels_modality_1 = labels_total_modality_1[
                first_fall_frame_modality_1 : last_fall_frame_modality_1 + 1
            ]
            fall_frame_indices_modality_1 = np.arange(first_fall_frame_modality_1, last_fall_frame_modality_1 + 1)
    # Video contains two fall
    elif transitions_modality_1 == transitions_modality_2 == 4:
        # Use helper function to equalise first fall, intermediate frames and second fall
        (
            fall_frames_modality_1,
            fall_frame_labels_modality_1,
            fall_frames_modality_2,
            fall_frame_labels_modality_2,
            fall_frame_indices_modality_1,
            fall_frame_indices_modality_2,
        ) = equalise_two_falls_helper(
            vid_total_modality_1,
            vid_total_modality_2,
            labels_total_modality_1,
            labels_total_modality_2,
            diff_modality_1,
            diff_modality_2,
        )

    # 3. Equalise frames after the fall event across both modalities

    if frames_after_fall_modality_1 >= frames_after_fall_modality_2:
        after_fall_frames_modality_1, after_fall_frame_labels_modality_1, after_fall_frame_indices_modality_1 = (
            equalise_frames_helper(
                last_fall_frame_modality_1 + 1,
                len(vid_total_modality_1) - 1,
                vid_total_modality_1,
                labels_total_modality_1,
                count=frames_after_fall_modality_2,
            )
        )
        after_fall_frames_modality_2 = vid_total_modality_2[last_fall_frame_modality_2 + 1 :]
        after_fall_frame_labels_modality_2 = labels_total_modality_2[last_fall_frame_modality_2 + 1 :]
        after_fall_frame_indices_modality_2 = np.arange(last_fall_frame_modality_2 + 1, len(vid_total_modality_2))
    else:
        after_fall_frames_modality_2, after_fall_frame_labels_modality_2, after_fall_frame_indices_modality_2 = (
            equalise_frames_helper(
                last_fall_frame_modality_2 + 1,
                len(vid_total_modality_2) - 1,
                vid_total_modality_2,
                labels_total_modality_2,
                count=frames_after_fall_modality_1,
            )
        )
        after_fall_frames_modality_1 = vid_total_modality_1[last_fall_frame_modality_1 + 1 :]
        after_fall_frame_labels_modality_1 = labels_total_modality_1[last_fall_frame_modality_1 + 1 :]
        after_fall_frame_indices_modality_1 = np.arange(last_fall_frame_modality_1 + 1, len(vid_total_modality_1))

    # 4. Concatenate the three slices for both the modalities

    vid_total_modality_1 = np.concatenate(
        (
            before_fall_frames_modality_1,
            fall_frames_modality_1,
            after_fall_frames_modality_1,
        ),
        axis=0,
    )
    labels_total_modality_1 = np.concatenate(
        (
            before_fall_frame_labels_modality_1,
            fall_frame_labels_modality_1,
            after_fall_frame_labels_modality_1,
        ),
        axis=0,
    )
    original_indices_modality_1 = np.concatenate(
        (
            before_fall_frame_indices_modality_1,
            fall_frame_indices_modality_1,
            after_fall_frame_indices_modality_1,
        ),
        axis=0,
    )

    vid_total_modality_2 = np.concatenate(
        (
            before_fall_frames_modality_2,
            fall_frames_modality_2,
            after_fall_frames_modality_2,
        ),
        axis=0,
    )
    labels_total_modality_2 = np.concatenate(
        (
            before_fall_frame_labels_modality_2,
            fall_frame_labels_modality_2,
            after_fall_frame_labels_modality_2,
        ),
        axis=0,
    )
    original_indices_modality_2 = np.concatenate(
        (
            before_fall_frame_indices_modality_2,
            fall_frame_indices_modality_2,
            after_fall_frame_indices_modality_2,
        ),
        axis=0,
    )

    ## Print Statements

    # fall_length_modality_1 = (labels_total_modality_1 == 1).sum()
    # fall_length_modality_2 = (labels_total_modality_2 == 1).sum()

    # print("\n*Modified Length - ", len(vid_total_modality_1), len(vid_total_modality_2))
    # print("*Modified Fall Frames - ", fall_length_modality_1, fall_length_modality_2)

    # if transitions_modality1 == transitions_modality2 == 2:
    #     first_fall_frame_modality_1 = (np.where(labels_total_modality_1 == 1)[0]).tolist()[0]
    #     last_fall_frame_modality_1 = (np.where(labels_total_modality_1 == 1)[0]).tolist()[-1]
    #     first_fall_frame_modality_2 = (np.where(labels_total_modality_2 == 1)[0]).tolist()[0]
    #     last_fall_frame_modality_2 = (np.where(labels_total_modality_2 == 1)[0]).tolist()[-1]
    #     print("*Start -", first_fall_frame_modality_1,"End -", last_fall_frame_modality_1,", Start -", first_fall_frame_modality_2,"End -", last_fall_frame_modality_2)  # fmt: skip
    # elif transitions_modality1 == transitions_modality2 == 4:
    #     diff_modality_1 = np.diff(labels_total_modality_1)
    #     diff_modality_2 = np.diff(labels_total_modality_2)

    #     fall_1_first_frame_modality_1 = (np.where(diff_modality_1 == 1)[0][0]) + 1
    #     fall_1_last_frame_modality_1 = np.where(diff_modality_1 == -1)[0][0]
    #     fall_2_first_frame_modality_1 = (np.where(diff_modality_1 == 1)[0][1]) + 1
    #     fall_2_last_frame_modality_1 = np.where(diff_modality_1 == -1)[0][1]

    #     fall_1_first_frame_modality_2 = (np.where(diff_modality_2 == 1)[0][0]) + 1
    #     fall_1_last_frame_modality_2 = np.where(diff_modality_2 == -1)[0][0]
    #     fall_2_first_frame_modality_2 = (np.where(diff_modality_2 == 1)[0][1]) + 1
    #     fall_2_last_frame_modality_2 = np.where(diff_modality_2 == -1)[0][1]

    #     print("*Modified Double Fall\n*Fall 1 : Start -", fall_1_first_frame_modality_1,"End -", fall_1_last_frame_modality_1,", Start -", fall_1_first_frame_modality_2,"End -", fall_1_last_frame_modality_2)  # fmt: skip
    #     print("*Fall 2 : Start -", fall_2_first_frame_modality_1,"End -", fall_2_last_frame_modality_1,", Start -", fall_2_first_frame_modality_2,"End -", fall_2_last_frame_modality_2)  # fmt: skip

    # print("------------------------------------------------")

    return (
        vid_total_modality_1,
        vid_total_modality_2,
        labels_total_modality_1,
        labels_total_modality_2,
        original_indices_modality_1,
        original_indices_modality_2,
    )


# # For using preprocessed images from h5py as input
# name = "Thermal_T3"
# path = "{}\Dataset\H5PY\{}_Data_set-{}-imgdim64x64.h5".format(project_directory, dataset_category, name)
# with h5py.File(path, "r") as hf:
#     data_dict = hf["{}/Processed/Split_by_video".format(name)]
#     # Any Fall directory
#     vid_total_fall = data_dict["Fall0"]["Data"][:]
#     labels_total_fall = data_dict["Fall0"]["Labels"][:]
#     fall_frame_extractor(vid_total_fall, labels_total_fall)
#     # Any ADL directory
#     vid_total_non_fall = data_dict["NonFall0"]["Data"][:]
#     labels_total_non_fall = data_dict["NonFall0"]["Labels"][:]
#     key_frame_extractor(vid_total_non_fall, labels_total_non_fall)
