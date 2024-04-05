import h5py
import os
import sys
import numpy as np
from torch.utils import data
import re
from io import StringIO
from parameters import (
    project_directory,
    dataset_directory,
    dataset_category,
    feature_extraction,
    background_subtraction,
    batch_size,
    ht,
    wd,
    anomaly_detection_model,
    test_size,
    metadata_set,
    pad_video,
    key_frame_extraction,
    key_frame_extraction_algorithm,
    key_frame_extraction_algorithms,
    data_augmentation,
    folders_to_be_augmented,
)
from Feature_Extraction.background_subtractor import perform_background_subtraction
from Frame_Utility.frame_utility import fall_frame_extractor, key_frame_extractor
from sklearn.model_selection import train_test_split
from Data_Augmentation.augmenter import augment_images


# Load the H5PY dataset into a pytorch dataset class.
def create_pytorch_dataset(name, dset, path, window_len, fair_compairson, stride, TOD="Both"):
    falls = []
    adl = []
    if fair_compairson == True:  #  Load specific lists of fall and non-fall video directories based on time of day.
        shared_adl_vids = np.loadtxt(
            "{}/Dataset/Metadata/shared_adl_vids_{}.txt".format(project_directory, metadata_set)
        ).astype(int)
        shared_fall_vids = np.loadtxt(
            "{}/Dataset/Metadata/shared_fall_vids_{}.txt".format(project_directory, metadata_set)
        ).astype(int)
        day_fall_vids = np.loadtxt(
            "{}/Dataset/Metadata/day_fall_vids_{}.txt".format(project_directory, metadata_set)
        ).astype(int)
        night_fall_vids = np.loadtxt(
            "{}/Dataset/Metadata/night_fall_vids_{}.txt".format(project_directory, metadata_set)
        ).astype(int)
        if TOD == "Day":
            tod_list = day_fall_vids
        if TOD == "Night":
            tod_list = night_fall_vids
        if TOD == "Both":
            tod_list = shared_fall_vids

        # print(shared_fall_vids)
        # create specfic list of fall folders and nonfall folders (Day or Night or Shared)
        for root, dirs, files in os.walk(
            "{}/Dataset/Fall-Data/{}/{}/Fall".format(dataset_directory, dataset_category, dset)
        ):
            for dir in dirs:
                x = re.findall("[0-9]+", dir)[0]
                if (int(x) in shared_fall_vids) and (int(x) in tod_list):
                    falls.append(dir)

        for root, dirs, files in os.walk(
            "{}/Dataset/Fall-Data/{}/{}/NonFall".format(dataset_directory, dataset_category, dset)
        ):
            for dir in dirs:
                x = re.findall("[0-9]+", dir)[0]
                if int(x) in shared_adl_vids:
                    adl.append(dir)
        # print(len(falls))
        # print(len(adl))
    elif fair_compairson == False:  # Load all fall and non-fall video directories.
        # create list of all fall and nonfall folders
        for root, dirs, files in os.walk(
            "{}/Dataset/Fall-Data/{}/{}/Fall".format(dataset_directory, dataset_category, dset)
        ):
            if len(dirs) > 0:
                falls.extend(dirs)
        for root, dirs, files in os.walk(
            "{}/Dataset/Fall-Data/{}/{}/NonFall".format(dataset_directory, dataset_category, dset)
        ):
            if len(dirs) > 0:
                adl.extend(dirs)
        # print(len(falls))
        # print(len(adl))

    # Sort the lists based on folder number (Based on the digits after 'Fall', 'NonFall')
    falls.sort(key=lambda item: int(item.split("Fall")[1]))
    adl.sort(key=lambda item: int(item.split("NonFall")[1]))

    x_data_fall = []  # Video data
    y_data_fall = []  # Label
    x_info_fall = []  # Directory name/number

    x_data_adl = []
    y_data_adl = []
    x_info_adl = []

    # For CNN class imbalance problem
    # Fall frames exists only in fall folder, Non fall frames can be in both fall and non fall folder
    # Non fall frames is not extracted from fall folder as it is from younger adults.
    total_fall_frames = total_non_fall_frames_from_adl = 0

    # path = "processed_data\data_set-{}-imgdim64x64.h5".format(name)

    # Loop through fall and non-fall directories, loading video data and labels into respective lists.
    with h5py.File(path, "r") as hf:
        data_dict = hf["{}/Processed/Split_by_video".format(name)]
        # print(data_dict.keys())
        for Fall_name in falls:
            try:
                vid_total = data_dict[Fall_name]["Data"][:]
                labels_total = data_dict[Fall_name]["Labels"][:]
                # print("{} - {}, {} ".format(Fall_name, len(vid_total), len(labels_total)))
                if len(vid_total) < 10:
                    continue

                if key_frame_extraction:
                    # For Autoencoders, only Fall folders will be used for testing. So both Fall and ADL frames should be extracted
                    if anomaly_detection_model:
                        vid_total, background_subtracted_key_frames, labels_total = key_frame_extractor(
                            vid_total, labels_total, modality=name
                        )
                    # For CNNs, Both Fall and ADL folders will be used for testing. So extract only Fall frames to minimise class imbalance
                    else:
                        # Fall frames from original video, corrresponding labels
                        vid_total, labels_total = fall_frame_extractor(vid_total, labels_total)
                        total_fall_frames = total_fall_frames + len(labels_total)
                    # print("{} - {}, {} ".format(Fall_name, len(vid_total), len(labels_total)))

                if feature_extraction:
                    if background_subtraction:
                        # If background subtraction algorithm was already used for key frame extraction, reuse that
                        if (
                            key_frame_extraction
                            & anomaly_detection_model
                            & (key_frame_extraction_algorithm == key_frame_extraction_algorithms[0])
                        ):
                            vid_total = background_subtracted_key_frames
                        else:
                            vid_total = perform_background_subtraction(vid_total)
                    # print("{} - {}".format(Fall_name, len(vid_total)))

                x_data_fall.append(vid_total)
                x_info_fall.append(Fall_name)  # [7:]
                y_data_fall.append(labels_total)
                # print("{} - {}, {} ".format(Fall_name, len(vid_total), len(labels_total)))
            except:
                print("Skipped", Fall_name)

            # Exit after 5 fall directories (For dev and debugging)
            # if len(x_data_fall) == 5:
            #     break

        for adl_name in adl:
            try:
                vid_total = data_dict[adl_name]["Data"][:]
                labels_total = data_dict[adl_name]["Labels"][:]
                # print("{} - {}, {} ".format(adl_name, len(vid_total), len(labels_total)))
                if len(vid_total) < 10:
                    continue

                if key_frame_extraction:
                    # For Both Autoencoders and CNNs.
                    # Key frames from original video, background subtracted key frames, corresponding labels
                    vid_total, background_subtracted_key_frames, labels_total = key_frame_extractor(
                        vid_total, labels_total, modality=name
                    )
                    total_non_fall_frames_from_adl = total_non_fall_frames_from_adl + len(labels_total)
                    # print("{} - {}, {} ".format(adl_name, len(vid_total), len(labels_total)))

                if feature_extraction:
                    if background_subtraction:
                        # If background subtraction algorithm was already used for key frame extraction, reuse that
                        if key_frame_extraction & (
                            key_frame_extraction_algorithm == key_frame_extraction_algorithms[0]
                        ):
                            vid_total = background_subtracted_key_frames
                        else:
                            vid_total = perform_background_subtraction(vid_total)
                    # print("{} - {}".format(adl_name, len(vid_total)))

                x_data_adl.append(vid_total)
                x_info_adl.append(adl_name)  # [7:]
                y_data_adl.append(labels_total)
                # print("{} - {}, {} ".format(adl_name, len(vid_total), len(labels_total)))
            except:
                print("Skipped", adl_name)

            # Exit after 5 adl directories (For dev and debugging)
            # if len(x_data_adl) == 5:
            #     break

        # Data Augmentation for ADL (Training Set for Autoencoder Model's)
        if data_augmentation:
            for adl_name in folders_to_be_augmented:
                adl_name = "NonFall" + adl_name
                try:
                    vid_total = data_dict[adl_name]["Data"][:]
                    labels_total = data_dict[adl_name]["Labels"][:]
                    # print("{} - {}, {} ".format(adl_name, len(vid_total), len(labels_total)))
                    if len(vid_total) < 10:
                        continue

                    # Augment before key frame and feature extraction.
                    vid_total = augment_images(vid_total)

                    if key_frame_extraction:
                        # For Both Autoencoders and CNNs.
                        # Key frames from original video, background subtracted key frames, corresponding labels
                        vid_total, background_subtracted_key_frames, labels_total = key_frame_extractor(
                            vid_total, labels_total, modality=name
                        )
                        total_non_fall_frames_from_adl = total_non_fall_frames_from_adl + len(labels_total)
                        # print("{} - {}, {} ".format(Fall_name, len(vid_total), len(labels_total)))

                    if feature_extraction:
                        if background_subtraction:
                            # If background subtraction algorithm was already used for key frame extraction, reuse that
                            if key_frame_extraction & (
                                key_frame_extraction_algorithm == key_frame_extraction_algorithms[0]
                            ):
                                vid_total = background_subtracted_key_frames
                            else:
                                vid_total = perform_background_subtraction(vid_total)
                        # print("{} - {}".format(adl_name, len(vid_total)))

                    x_data_adl.append(vid_total)
                    x_info_adl.append(adl_name)  # [7:]
                    y_data_adl.append(labels_total)
                    # print("{} - {}, {} ".format(adl_name, len(vid_total), len(labels_total)))
                except:
                    print("Skipped", adl_name)
                # Exit after 5 adl directories (For dev and debugging)
                # if len(x_data_adl) == 5:
                #     break

    # pdb.set_trace()
    # %%    temp_df = my_data.loc[my_data["Video"] == int(fall), "ToD"]

    # ----------------------------------------------------------------------------
    # *** PREPARING DATASET LOADER ***
    # ----------------------------------------------------------------------------

    # 1) Need a ADL loader and a Fall Loader

    class Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, labels, data, window):
            "Initialization"
            self.labels = labels
            self.data = data
            self.window = window

        def __len__(self):
            "Denotes the total number of samples"
            return len(self.data)

        def __getitem__(self, index):
            "Generates one sample of data"
            # prepare lists to dynamically fill with windows
            X_list = []
            Y_list = []
            # load a single video to chop up into windows
            ind_vid = self.data[index]
            ind_label = self.labels[index]
            # loop through each frame of the video (stopping window length short)
            for i in range(len(ind_vid) - self.window):
                # select the current window of the video
                X = ind_vid[i : i + self.window]
                y = ind_label[i : i + self.window]
                # add the current window the list of windows
                X_list.append(X)
                Y_list.append(y)
            # convert lists into arrays with proper size
            X = np.vstack(X_list)
            X = np.reshape(X_list, (len(ind_vid) - self.window, self.window, ht, wd))
            y = np.vstack(Y_list).T
            # X should be (window-length, 64, 64, # of windows w/in video) array
            # ex. (8, 64, 64, 192) for a 200 frame video and window size of 8
            # y is array (8, # of windows w/in video)
            return X, y

    # print(len(x_data_fall))
    # print(len(x_data_adl))
    # print(total_fall_frames, total_non_fall_frames_from_adl)

    if anomaly_detection_model:
        Test_Dataset = Dataset(y_data_fall, x_data_fall, window=window_len)  # Parameters - Labels, Data, Window length
        test_dataloader = data.DataLoader(Test_Dataset, batch_size)

        Train_Dataset = Dataset(y_data_adl, x_data_adl, window=window_len)
        train_dataloader = data.DataLoader(Train_Dataset, batch_size)

        return (Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)
    else:
        # Combine Fall and ADL arrays
        combined_x_data = x_data_fall + x_data_adl  # Data
        combined_y_data = y_data_fall + y_data_adl  # Labels

        x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
            combined_x_data, combined_y_data, test_size=test_size, random_state=42
        )  # Parameters - Data, Labels, Split Ratio, Random Seed

        # Create separate datasets and dataloaders for training and testing
        Test_Dataset = Dataset(y_data_test, x_data_test, window=window_len)
        test_dataloader = data.DataLoader(Test_Dataset, batch_size)

        Train_Dataset = Dataset(y_data_train, x_data_train, window=window_len)
        train_dataloader = data.DataLoader(Train_Dataset, batch_size)

        return (Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)


def create_multimodal_pytorch_dataset(names, dsets, paths, window_len, fair_compairson, stride):

    def load_data(name, dset, path, fair_compairson):
        falls = []
        adl = []
        if fair_compairson == True:
            shared_adl_vids = np.loadtxt(
                "{}/Dataset/Metadata/shared_adl_vids_{}.txt".format(project_directory, metadata_set)
            ).astype(int)
            shared_fall_vids = np.loadtxt(
                "{}/Dataset/Metadata/shared_fall_vids_{}.txt".format(project_directory, metadata_set)
            ).astype(int)
            # create list of all fall and nonfall folders
            for root, dirs, files in os.walk(
                "{}/Dataset/Fall-Data/{}/{}/Fall".format(dataset_directory, dataset_category, dset)
            ):
                for dir in dirs:
                    x = re.findall("[0-9]+", dir)[0]
                    if int(x) in shared_fall_vids:
                        falls.append(dir)

            for root, dirs, files in os.walk(
                "{}/Dataset/Fall-Data/{}/{}/NonFall".format(dataset_directory, dataset_category, dset)
            ):
                for dir in dirs:
                    x = re.findall("[0-9]+", dir)[0]
                    if int(x) in shared_adl_vids:
                        adl.append(dir)

        elif fair_compairson == False:
            # create list of all fall and nonfall folders
            for root, dirs, files in os.walk(
                "{}/Dataset/Fall-Data/{}/{}/Fall".format(dataset_directory, dataset_category, dset)
            ):
                if len(dirs) > 0:
                    falls.extend(dirs)
            for root, dirs, files in os.walk(
                "{}/Dataset/Fall-Data/{}/{}/NonFall".format(dataset_directory, dataset_category, dset)
            ):
                if len(dirs) > 0:
                    adl.extend(dirs)

        x_data_fall = []  # Video data
        y_data_fall = []  # Label
        x_info_fall = []  # Directory name/number

        x_data_adl = []
        y_data_adl = []
        x_info_adl = []

        # path = "processed_data\data_set-{}-imgdim64x64.h5".format(name)

        # load in images of falls
        with h5py.File(path, "r") as hf:
            data_dict = hf["{}/Processed/Split_by_video".format(name)]
            # print(data_dict.keys())

            for Fall_name in falls:
                try:
                    vid_total = data_dict[Fall_name]["Data"][:]
                    if len(vid_total) < 10:
                        continue
                    x_data_fall.append(vid_total)
                    x_info_fall.append(Fall_name)  # [4:]
                    labels_total = data_dict[Fall_name]["Labels"][:]
                    y_data_fall.append(labels_total)
                except:
                    print("Skipped", Fall_name)

                # Exit after 5 fall directories (For dev and debugging)
                # if len(x_data_fall) == 5:
                #     break

            for adl_name in adl:
                try:
                    vid_total = data_dict[adl_name]["Data"][:]
                    if len(vid_total) < 10:
                        continue
                    x_data_adl.append(vid_total)
                    x_info_adl.append(adl_name)  # [7:]
                    labels_total = data_dict[adl_name]["Labels"][:]
                    y_data_adl.append(labels_total)
                except:
                    print("Skipped", adl_name)

                # Exit after 5 fall directories (For dev and debugging)
                # if len(x_data_adl) == 5:
                #     break

        """ 
        # get matching day/night label from falls
        labels_dir = "D:/{}/".format(dset) + "Labels.csv"
        my_data = pd.read_csv(labels_dir)
        # sorting by first name
        my_data.sort_values("Video", inplace=True)
        my_data.drop_duplicates(subset="Video", keep="first", inplace=True)
        print(my_data.head())
        """
        return (y_data_fall, x_data_fall, x_data_adl, y_data_adl)

    # ----------------------------------------------------------------------------
    # *** PREPARING DATASET LOADER ***
    # ----------------------------------------------------------------------------

    # 1) Need a ADL loader and a Fall Loader

    y_data_falls = []
    x_data_falls = []
    y_data_adls = []
    x_data_adls = []

    for i in range(len(dsets)):
        path = paths[i]
        print("Modality {} - {}".format(i + 1, dsets[i]))
        y_data_fall, x_data_fall, x_data_adl, y_data_adl = load_data(names[i], dsets[i], path, fair_compairson)
        # print(len(y_data_fall), len(x_data_fall), len(y_data_adl), len(x_data_adl))
        print("Non Falls - {}, Falls - {}\n".format(len(x_data_adl), len(x_data_fall)))
        y_data_falls.append(y_data_fall)
        x_data_falls.append(x_data_fall)
        y_data_adls.append(y_data_adl)
        x_data_adls.append(x_data_adl)
        del y_data_fall
        del x_data_fall
        del y_data_adl
        del x_data_adl

    """' 
    def re_arrange_data(x_data, y_data):
        #(3, 8, 64, 64, 192) for a 200 frame video and window size of 8 and 3 modalities
        #loop through each modality 
        X_modality_list = []
        y_modality_list = []
        # loop through each frame of the video (stopping window length short)
        for j in range(0, len(x_data)):
            print(j)
            X_video_list = []
            y_label_list = []
            for vid in range(len(x_data[j])):
                print(vid)
                X_list = []
                Y_list = []
                #find shortest modality
                max_length = 0 
                for k in range(len(x_data)):
                    if len(x_data[k][vid]) > max_length:
                        max_length = len(x_data[k][vid]) 
                for j in range(len(x_data)):
                    while len(x_data[k][vid]) < max_length:

                        x_data[k][vid] = np.pad(x_data[k][vid], [(0, 1), (0, 0), (0, 0), (0, 0)], 'mean')#, (0,0)
                        y_data[k][vid] = np.append(y_data[k][vid], 0) #, (0,0)

                for i in range(0, len(y_data[j][vid]) - window_len):
                    # select the current window of the video
                    X = x_data[j][vid][i : i + window_len][:]
                    y = y_data[j][vid][i : i + window_len]
                    # add the current window the list of windows
                    X_list.append(X)
                    Y_list.append(y)
                X_video_list.append(np.asarray(X_list))
                y_label_list.append(np.asarray(Y_list))
                
            X_modality_list.append(X_video_list)
            y_modality_list.append(y_label_list)
        return(X_modality_list, y_modality_list)

    X_modality_list, y_modality_list = re_arrange_data(x_data_falls, y_data_falls)
    """

    class Multi_Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, labels, datas, window):
            "Initialization"
            self.labels = labels
            self.datas = datas
            self.window = window

        def __len__(self):
            "Denotes the total number of modalities"
            return len(self.datas[0])

        def __getitem__(self, index):
            "Generates one sample of data"

            x_data = self.datas
            y_data = self.labels

            vid = index
            X_modality_list = []  # Data (all modality)
            y_modality_list = []  # Labels (all modality)

            # Find the maximum length of frames across all modalities for this video
            max_length = 0
            min_length = sys.maxsize
            # For each modality
            for k in range(len(x_data)):
                if len(x_data[k][vid]) > max_length:
                    max_length = len(x_data[k][vid])
                if len(x_data[k][vid]) < min_length:
                    min_length = len(x_data[k][vid])

            # print(len(x_data[0][vid]), len(x_data[1][vid]))

            if pad_video:
                # For each modality, The video will be padded to match the modality with the maximum length video
                for k in range(len(x_data)):
                    while len(x_data[k][vid]) < max_length:
                        # Pads the video with the mean value of its existing elements until max_length is reached
                        x_data[k][vid] = np.pad(x_data[k][vid], [(0, 1), (0, 0), (0, 0), (0, 0)], "mean")  # , (0,0)
                        y_data[k][vid] = np.append(y_data[k][vid], 0)  # , (0,0)
            else:
                # For each modality, The video will be trimmed to match the modality with the minimum length video
                for k in range(len(x_data)):
                    if len(x_data[k][vid]) > min_length:
                        x_data[k][vid] = x_data[k][vid][:min_length]
                        y_data[k][vid] = y_data[k][vid][:min_length]

            # print(len(x_data[0][vid]), len(x_data[1][vid]))

            # For each modality
            for k in range(len(x_data)):
                X_list = []  # Windowed Data for this video
                Y_list = []  # Windowed Labels for this video
                # create windows
                # loop through each frame of the video (stopping window length short)
                for i in range(0, len(y_data[k][vid]) - window_len):
                    # select the current window of the video
                    X = x_data[k][vid][i : i + window_len][:]
                    y = y_data[k][vid][i : i + window_len]
                    # add the current window to the list of windows
                    X_list.append(X)
                    Y_list.append(y)

                X_modality_list.append(np.asarray(X_list))  # Windowed data of all modalities is saved one by one
                y_modality_list.append(np.asarray(Y_list))  # Windowed labels of all modalities is saved one by one

            X = np.squeeze(np.stack(X_modality_list))
            y = np.squeeze(np.stack(y_modality_list))
            # X should be (modalities, # of windows w/in video, window-length, 64, 64) array
            # ex - (2, 819, 8, 64, 64)
            # y should be (modalities, # of windows w/in video, window-length) array
            # ex - (2, 819, 8)
            return (X, y)
            """
            mod_vid = [] 
            mod_labels = []
            for j in range(len(self.datas)):
                video = self.datas[j][index]
                label = self.labels[j][index]
                mod_vid.append(np.squeeze(video))
                mod_labels.append(np.squeeze(label))
            X = np.squeeze(np.stack(mod_vid))
            y = np.squeeze(np.stack(mod_labels))
            """
            # X should be (modalities, window-length, 64, 64, # of windows w/in video) array
            # ex. (3, 8, 64, 64, 192) for a 200 frame video and window size of 8 and 3 modalities
            # y is array (8, # of windows w/in video)
            return X, y

    # X_modality_list, y_modality_list = re_arrange_data(x_data_adls, y_data_adls)

    Test_Dataset = Multi_Dataset(y_data_falls, x_data_falls, window=window_len)
    test_dataloader = data.DataLoader(Test_Dataset, batch_size)

    Train_Dataset = Multi_Dataset(y_data_adls, x_data_adls, window=window_len)
    train_dataloader = data.DataLoader(Train_Dataset, batch_size)

    return (Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)
