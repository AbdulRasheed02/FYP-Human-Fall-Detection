import h5py
import os
import numpy as np
from torch.utils import data
import re
from io import StringIO
from parameters import (
    project_directory,
    dataset_directory,
    feature_extraction,
    background_subtraction,
    batch_size,
    ht,
    wd,
    anomaly_detection_model,
    test_size,
    metadata_set,
)
from Feature_Extraction.background_subtractor import perform_background_subtraction
from sklearn.model_selection import train_test_split


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
        for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/Fall".format(dataset_directory, dset)):
            for dir in dirs:
                x = re.findall("[0-9]+", dir)[0]
                if (int(x) in shared_fall_vids) and (int(x) in tod_list):
                    falls.append(dir)

        for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/NonFall".format(dataset_directory, dset)):
            for dir in dirs:
                x = re.findall("[0-9]+", dir)[0]
                if int(x) in shared_adl_vids:
                    adl.append(dir)
        # print(falls)
        # print(adl)
    elif fair_compairson == False:  # Load all fall and non-fall video directories.
        # create list of all fall and nonfall folders
        for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/Fall".format(dataset_directory, dset)):
            if len(dirs) > 0:
                falls.extend(dirs)
        for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/NonFall".format(dataset_directory, dset)):
            if len(dirs) > 0:
                adl.extend(dirs)
        # print(falls)
        # print(adl)

    x_data_fall = []  # Video data
    y_data_fall = []  # Label
    x_info_fall = []  # Directory name/number

    x_data_adl = []
    y_data_adl = []
    x_info_adl = []

    # path = "processed_data\data_set-{}-imgdim64x64.h5".format(name)

    # Loop through fall and non-fall directories, loading video data and labels into respective lists.
    with h5py.File(path, "r") as hf:
        data_dict = hf["{}/Processed/Split_by_video".format(name)]
        # print(data_dict.keys())
        for Fall_name in falls:
            try:
                vid_total = data_dict[Fall_name]["Data"][:]
                # print("{} - {} ".format(Fall_name, len(vid_total)))
                if len(vid_total) < 10:
                    continue

                if feature_extraction:
                    feature_extracted_vid_total = []
                    if background_subtraction:
                        feature_extracted_vid_total = perform_background_subtraction(vid_total)
                    # print("{} - {}".format(Fall_name, len(feature_extracted_vid_total)))
                    x_data_fall.append(feature_extracted_vid_total)
                else:
                    x_data_fall.append(vid_total)

                x_info_fall.append(Fall_name)  # [7:]
                labels_total = data_dict[Fall_name]["Labels"][:]
                y_data_fall.append(labels_total)
            except:
                print("Skipped", Fall_name)

            # Exit after 5 fall directories (For dev and debugging)
            if len(x_data_fall) == 5:
                break

        for adl_name in adl:
            try:
                vid_total = data_dict[adl_name]["Data"][:]
                # print("{} - {} ".format(adl_name, len(vid_total)))
                if len(vid_total) < 10:
                    continue

                if feature_extraction:
                    feature_extracted_vid_total = []
                    if background_subtraction:
                        feature_extracted_vid_total = perform_background_subtraction(vid_total)
                    # print("{} - {}".format(adl_name, len(feature_extracted_vid_total)))
                    x_data_adl.append(feature_extracted_vid_total)
                else:
                    x_data_adl.append(vid_total)

                x_info_adl.append(adl_name)  # [7:]
                labels_total = data_dict[adl_name]["Labels"][:]
                y_data_adl.append(labels_total)
            except:
                print("Skipped", adl_name)

            # Exit after 5 adl directories (For dev and debugging)
            if len(x_data_adl) == 5:
                break

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


def create_multimodal_pytorch_dataset(names, dsets, window_len, fair_compairson, stride):

    class Multi_Dataset(data.Dataset):
        "Characterizes a dataset for PyTorch"

        def __init__(self, labels, datas, window):
            "Initialization"
            self.labels = labels
            self.datas = datas
            self.window = window

        def __len__(self):
            "Denotes the total number of modalities"
            print(len(self.datas[0]))
            return len(self.datas[0])

        def __getitem__(self, index):
            "Generates one sample of data"

            x_data = self.datas
            y_data = self.labels

            vid = index
            X_modality_list = []
            y_modality_list = []
            # loop through each frame of the video (stopping window length short)
            for j in range(len(x_data)):
                X_list = []
                Y_list = []
                # have to make arrays same size
                max_length = 0
                for k in range(len(x_data)):
                    if len(x_data[k][vid]) > max_length:
                        max_length = len(x_data[k][vid])
                for k in range(len(x_data)):
                    while len(x_data[k][vid]) < max_length:
                        x_data[k][vid] = np.pad(x_data[k][vid], [(0, 1), (0, 0), (0, 0), (0, 0)], "mean")  # , (0,0)
                        y_data[k][vid] = np.append(y_data[k][vid], 0)  # , (0,0)
                # create windows

                for i in range(0, len(y_data[j][vid]) - window_len):
                    # select the current window of the video
                    X = x_data[j][vid][i : i + window_len][:]
                    y = y_data[j][vid][i : i + window_len]
                    # add the current window the list of windows
                    X_list.append(X)
                    Y_list.append(y)
                # save videos into list
                X_modality_list.append(np.asarray(X_list))
                y_modality_list.append(np.asarray(Y_list))
            X = np.squeeze(np.stack(X_modality_list))
            y = np.squeeze(np.stack(y_modality_list))
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

    def load_data(name, dset, path, fair_compairson):
        print(name, dset, path)
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
            for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/Fall".format(dataset_directory, dset)):
                for dir in dirs:
                    x = re.findall("[0-9]+", dir)[0]
                    if int(x) in shared_fall_vids:
                        falls.append(dir)

            for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/NonFall".format(dataset_directory, dset)):
                for dir in dirs:
                    x = re.findall("[0-9]+", dir)[0]
                    if int(x) in shared_adl_vids:
                        adl.append(dir)

        elif fair_compairson == False:
            # create list of all fall and nonfall folders
            for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/Fall".format(dataset_directory, dset)):
                if len(dirs) > 0:
                    falls.extend(dirs)
            for root, dirs, files in os.walk("{}/Dataset/Fall-Data/{}/NonFall".format(dataset_directory, dset)):
                if len(dirs) > 0:
                    adl.extend(dirs)

        x_data_fall = []
        y_data_fall = []
        x_info_fall = []

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

            for adl_name in adl:
                try:
                    vid_total = data_dict[adl_name]["Data"][:]
                    if len(vid_total) < 10:
                        print(adl_name)
                        continue
                    x_data_adl.append(vid_total)
                    x_info_adl.append(adl_name)  # [7:]
                    labels_total = data_dict[adl_name]["Labels"][:]
                    y_data_adl.append(labels_total)
                except:
                    print("Skipped", adl_name)

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
        path = "{}\H5PY\Data_set-{}-imgdim64x64.h5".format(project_directory, names[i])
        print("loading", names[i])
        y_data_fall, x_data_fall, x_data_adl, y_data_adl = load_data(names[i], dsets[i], path, fair_compairson)
        y_data_falls.append(y_data_fall)
        x_data_falls.append(x_data_fall)
        y_data_adls.append(y_data_adl)
        x_data_adls.append(x_data_adl)
        print(len(y_data_fall))
        print(len(x_data_fall))
        print(len(y_data_adl))
        print(len(x_data_adl))
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

    Test_Dataset = Multi_Dataset(y_data_falls, x_data_falls, window=window_len)
    test_dataloader = data.DataLoader(Test_Dataset, batch_size)

    # X_modality_list, y_modality_list = re_arrange_data(x_data_adls, y_data_adls)

    Train_Dataset = Multi_Dataset(y_data_adls, x_data_adls, window=window_len)
    train_dataloader = data.DataLoader(Train_Dataset, batch_size)

    return (Test_Dataset, test_dataloader, Train_Dataset, train_dataloader)
