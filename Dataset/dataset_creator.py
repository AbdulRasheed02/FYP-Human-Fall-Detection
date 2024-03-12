import os
import glob

import numpy as np
import cv2
import h5py
import re
import sys

# Importing constants from parameters.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from Code.parameters import project_directory, dataset_directory, ht, wd

sys.path.remove(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

"""
This file will create an h5py file of your dataset.       
This is a compressed version of your data that will be read faster. 
In order for this to work you must have your files organized in the following format.   
Falls and Non Falls
- numbered folders containing the video broken down into images
- (ex. Fall1, Fall2, Fall3 ... Fall35... etc) 
folder_location is the name of the folder that contains this dataset 
dset is going to be the compressed h5py file name of your dataset
"""


# Function definition to get lists of directories containing Fall and NonFall videos
def get_dir_lists(dset, folder_location):
    """
    Parameters:
        dset: h5py dataset name
        folder_location: path to the dataset folder
    Returns:
        vid_dir_list_ADL: list of directories containing NonFall videos
        vid_dir_list_Fall: list of directories containing Fall videos
    """

    path_Fall = folder_location + "\\Fall\\"
    path_ADL = folder_location + "\\NonFall\\"
    vid_dir_list_Fall = glob.glob(path_Fall + "Fall*")
    vid_dir_list_ADL = glob.glob(path_ADL + "NonFall*")

    return vid_dir_list_ADL, vid_dir_list_Fall


# Function definition to initialize the creation of an h5py file for the dataset
def init_videos(img_width, img_height, raw, sort, fill_depth, dset, folder_location):
    """
    Creates or overwrites h5py group corresponding to root_path (in body), for the h5py file located at
    'N:/...../FallDetection/Dataset/H5PY/Data_set-{}-imgdim{}x{}.h5'.format(dset, img_width, img_height).

    Parameters:
        img_width: width of images
        img_height: height of images
        raw: boolean flag indicating whether to process the data or not
        sort: boolean flag indicating whether to sort the frames or not
        fill_depth: boolean flag indicating whether to fill missing pixels for depth images or not
        dset: h5py dataset name
        folder_location: path to the dataset folder
    """

    path = "{}\\Dataset\H5PY\\Data_set-{}-imgdim{}x{}.h5".format(project_directory, dset, img_width, img_height)

    # Retrieves lists of directories containing Fall and NonFall videos.
    vid_dir_list_0, vid_dir_list_1 = get_dir_lists(dset, folder_location)
    print(vid_dir_list_0, dset, folder_location)
    print(len(vid_dir_list_0))
    print(len(vid_dir_list_1))

    if len(vid_dir_list_0) == 0 and len(vid_dir_list_1) == 0:
        print("No videos found, make sure video files are placed in Fall-Data folder, terminating...")
        sys.exit()

    if raw == False:
        root_path = dset + "/Processed/Split_by_video"
    else:
        root_path = dset + "/Raw/Split_by_video"

    print("Creating data at root_path", root_path)

    def init_videos_helper(root_path):  # Nested to keep scope
        with h5py.File(path, "w") as hf:
            # root_sub = root.create_group('Split_by_video')
            root = hf.create_group(root_path)

            # Intialising and storing Fall Videos in H5PY file
            for vid_dir in vid_dir_list_1:
                init_vid(
                    vid_dir=vid_dir,
                    vid_class=1,
                    img_width=img_width,
                    img_height=img_height,
                    hf=root,
                    raw=raw,
                    sort=sort,
                    fill_depth=fill_depth,
                    dset=dset,
                )

            # Intialising and storing ADL Videos in H5PY file
            for vid_dir in vid_dir_list_0:
                init_vid(
                    vid_dir=vid_dir,
                    vid_class=0,
                    img_width=img_width,
                    img_height=img_height,
                    hf=root,
                    raw=raw,
                    sort=sort,
                    fill_depth=fill_depth,
                    dset=dset,
                )

    if os.path.isfile(path):  # H5PY file already exists
        print("Going down other tree")
        hf = h5py.File(path, "w")
        if root_path in hf:  # Split_by_video group already exists.
            print("video h5py file exists, deleting old group {}, creating new".format(root_path))
            del hf[root_path]
            hf.close()
            init_videos_helper(root_path)
        else:  # Split_by_video group does not exist``.
            print("File exists, but no group for this data set; initializing..")
            hf.close()
            init_videos_helper(root_path)

    else:  # not initialized
        print("No data file exists yet; initializing")

        init_videos_helper(root_path)


# Helper function to initialize a single video within the h5py file
def init_vid(vid_dir, vid_class, img_width, img_height, hf, raw, sort, fill_depth, dset):
    """
    helper function for init_videos. Initialzies a single video.
    Parameters:
        vid_dir: directory containing frames of the video (Fall0, Fall1 etc.)
        vid_class: class label of the video (1 for Fall, 0 for NonFall)
        img_width: width of images
        img_height: height of images
        hf: h5py group where the video data will be stored
        raw: boolean flag indicating whether to process the data or not
        sort: boolean flag indicating whether to sort the frames or not
        fill_depth: boolean flag indicating whether to fill missing pixels for depth images or not
        dset: h5py dataset name
    """
    vid_dir_name = os.path.basename(vid_dir)
    m = re.search(r"\d+$", vid_dir_name)  # Extract 0 from Fall0, 1 from Fall1 etc.
    fall_number = int(m.group())
    import pandas as pd

    my_data = pd.read_csv(folder_location + "/Labels.csv")
    current_vid = my_data[my_data.Video == fall_number]
    if (len(current_vid) == 0) & (len(vid_dir_name) < 8):
        print("Skipping {} as it does not contain a fall".format(vid_dir_name))
        return

    print("-----------------------")
    print("initializing vid at", vid_dir_name, folder_location)

    fpath = vid_dir

    data = create_img_data_set(fpath, img_width, img_height, raw, sort, fill_depth, dset)

    print("Creating at", vid_dir_name)
    grp = hf.create_group(vid_dir_name)
    labels = np.zeros(
        len(data)
    )  # Initializes an array of zeros with the same length as the number of frames in the video.
    if vid_class == 1:
        labels = get_fall_indices(vid_dir_name, labels, dset)
        print(np.unique(labels, return_counts=True))
    else:
        print("Non fall thus labels are 0ed")

    # Stores the labels and image data in the h5py group for the current video.
    grp["Labels"] = labels
    grp["Data"] = data


# Function to assign labels to frames indicating the occurrence of falls
def get_fall_indices(Fall_name, labels, dset):
    """
    Parameters:
        Fall_name: name of the fall video
        labels: numpy array containing labels for each frame
        dset: h5py dataset name
    Returns:
        labels: numpy array with assigned labels
    """

    m = re.search(r"\d+$", Fall_name)
    fall_number = int(m.group())

    import pandas as pd

    my_data = pd.read_csv(folder_location + "/Labels.csv")

    current_vid = my_data[my_data.Video == fall_number]
    print(current_vid)
    if len(current_vid) == 0:
        return

    if len(current_vid) == 1:
        print("Single Fall")
        labels[int(current_vid.Start.iloc[0]) : int(current_vid.Stop.iloc[0])] = 1
    else:
        print("Two Falls")
        labels[int(current_vid.Start.iloc[0]) : int(current_vid.Stop.iloc[0])] = 1
        labels[int(current_vid.Start.iloc[1]) : int(current_vid.Stop.iloc[1])] = 1

    return labels


# Function to create a dataset of images from a directory
def create_img_data_set(fpath, ht, wd, raw=False, sort=True, fill_depth=False, dset="Thermal"):
    """
    Creates data set of all images located at fpath. Sorts images
    Parameters:
        fpath: path to the directory containing images
        ht: height of images
        wd: width of images
        raw: boolean flag indicating whether to process the data or not (default - False)
        sort: boolean flag indicating whether to sort the frames or not (default - True)
        fill_depth: boolean flag indicating whether to fill missing pixels for depth images or not (default - False)
        dset: h5py dataset name (default - Thermal)
    Returns:
        data: numpy array containing the images

    """

    # Note - raw, sort flags are not used. Data is always sorted and processed.

    # print('gathering data at', fpath)
    fpath = fpath.replace("\\", "/")
    # print(fpath+'/*.png')
    frames = glob.glob(fpath + "/*.jpg") + glob.glob(fpath + "/*.png")
    frames.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    # print("\n".join(frames)) #Use this to check if sorted

    # NumPy array 'data' with zeros. The shape of the array is determined by the number of frames, height (ht), width (wd), and a depth of 1.
    data = np.zeros((frames.__len__(), ht, wd, 1))

    # Loop iterates through each frame (x) in the sorted list of frames.
    for x, i in zip(frames, range(0, frames.__len__())):
        # print(x,i)
        img = cv2.imread(x, 0)  # Use this for RGB to Greyscale
        if fill_depth == True:
            thresh, maxval = 20, 255
            th, im_th = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY_INV)
            # print(np.amax(im_th), np.amin(im_th))
            mask = im_th
            dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)  # paints non-zero pixels
            img = dst
        try:
            img = cv2.resize(img, (ht, wd))  # resize
            img = img.reshape(ht, wd, 1)

            img = img - np.mean(img)  # Mean centering
            img = img.astype("float32") / 255  # rescaling

            data[i, :, :, :] = img  # Assigning the processed image to the corresponding position
        except Exception as e:
            print(str(e))

    print("data.shape", data.shape)
    return data


# Function to flip windowed data (sequence of frames) horizontally.
# Function is not used currently.
def flip_windowed_arr(windowed_data):
    """
    Parameters:
        windowed_data: numpy array containing windowed data
    Returns:
        flipped_data_windowed: numpy array with horizontally flipped windowed data
    """

    win_len = windowed_data.shape[1]
    flattened_dim = np.prod(windowed_data.shape[2:])
    # print(flattened_dim)
    flipped_data_windowed = np.zeros((len(windowed_data), win_len, flattened_dim))  # Array of windows
    print(flipped_data_windowed.shape)

    for win_idx in range(len(windowed_data)):  # Iterate over each window
        window = windowed_data[win_idx]
        flip_win = np.zeros((win_len, flattened_dim))

        for im_idx in range(len(window)):  # Iterate over each frame in a window
            im = window[im_idx]
            hor_flip_im = cv2.flip(im, 1)  # Horizontal flipping
            # print(hor_flip_im.shape)
            # print(flip_win[im_idx].shape)

            flip_win[im_idx] = hor_flip_im.reshape(flattened_dim)

        flipped_data_windowed[win_idx] = flip_win

    return flipped_data_windowed


# Directory names of the raw dataset from the Fall-Data folder
# modalities = ['Thermal','ONI_IR','IP']
modalities = ["Thermal"]

# Name of the H5PY files to be created
# dsets = ['Thermal_T3','ONI_IR_T','IP_T']
dsets = ["Thermal_T3"]

for i in range(len(modalities)):
    # location of were your dataset is stored
    modality = modalities[i]
    dset = dsets[i]
    folder_location = "{}\Dataset\Fall-Data\{}".format(dataset_directory, modality)
    print(folder_location)
    print(modality)

    img_width = wd
    img_height = ht
    raw = False
    sort = True

    if modality == "ONI_Depth":
        fill_depth = True
    else:
        fill_depth = False

    init_videos(img_width, img_height, raw, sort, fill_depth, dset, folder_location)
