import json
import math
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical


def normalization(array):

    norm_arr = (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))

    return norm_arr


def load_opt(filepath):

    with open(filepath, "r") as f:
        return json.load(f)


def load_brain_with_vol(dir_path, dir_list, opt):

    image_count = opt["scan"]["image_count"]
    use_hipp = opt["scan"]["use_hipp_vol"]
    use_icv = opt["scan"]["use_icv_adj_hipp"]
    use_asym = opt["scan"]["use_icv_adj_hipp"]
    w = opt["scan"]["size_w"]
    h = opt["scan"]["size_h"]
    l_ind = opt["scan"]["l_r_indices"][0]
    r_ind = opt["scan"]["l_r_indices"][1]
    i_ind = opt["scan"]["i_s_indices"][0]
    s_ind = opt["scan"]["i_s_indices"][1]
    p_ind = opt["scan"]["p_a_indices"][0]
    a_ind = opt["scan"]["p_a_indices"][1]
    offset = opt["scan"]["slice_offset"]

    extra = np.sum(np.array([use_hipp, use_icv, use_asym]))
    total_slices = image_count + extra

    filenames = np.empty(shape=(0, 1), dtype=str)
    brains_comb = np.empty(shape=(0, w, total_slices, h))
    vol_vect = np.empty(shape=(0, 2))

    for d in dir_list:

        dir_path_full = dir_path + d
        print(dir_path_full)

        filename = ""
        num_files = 0
        brains = np.zeros(shape=(num_files, w, total_slices, h))  # saves all the brains
        vols = np.zeros(shape=(num_files, 2))

        for (root, dirs, files) in os.walk(dir_path_full):

            num_files = len(files)  # get number of files in the dir

            for i in range(num_files):
                filename = files[i]
                img = nib.load(dir_path_full + "/" + files[i]).get_fdata()  # load the subject brain
                brains[i, :, :image_count, :] = img[l_ind:r_ind, p_ind:a_ind:offset, i_ind:s_ind]

        brains[brains == 0] = np.nan
        norm_brains = [normalization(i) for i in brains[:, :, :image_count, :]]
        norm_brains = np.array(norm_brains)
        norm_brains[np.isnan(norm_brains)] = 0

        if extra > 0:

            # Load volumes
            file_path_vol = dir_path + '../stats/icv_vol_' + d + '.csv'
            print(file_path_vol)

            df_vol = pd.read_csv(file_path_vol)
            hipp_l = df_vol.iloc[:, 0]
            hipp_r = df_vol.iloc[:, 1]
            icv = df_vol.iloc[:, 2]
            asym = df_vol.iloc[:, 3]
            s = image_count

            for i in range(num_files):

                slice1 = np.full((w, h), hipp_l[i])
                slice2 = np.full((w, h), hipp_r[i])

                if use_hipp:
                    norm_brains[i, :, s, :] = slice1
                    norm_brains[i, :, s+1, :] = slice2
                    s += 2

                if use_icv:
                    slice3 = slice1 / icv[i]
                    slice4 = slice2 / icv[i]
                    norm_brains[i, :, s, :] = slice3
                    norm_brains[i, :, s+1, :] = slice4
                    s += 2

                if use_asym:
                    slice5 = np.full((w, h), asym[i])
                    norm_brains[i, :, s, :] = slice5
                    s += 2

                vols[i, 0] = hipp_l[i]
                vols[i, 1] = hipp_r[i]

        filenames = np.concatenate((filenames, filename), axis=0)
        brains_comb = np.concatenate((brains_comb, norm_brains), axis=0)
        vol_vect = np.concatenate((vol_vect, vols), axis=0)

    return filenames, brains_comb, vol_vect


def load_img(brains, side, opt, img_slice):

    w = opt["scan"]["size_w"]
    h = opt["scan"]["size_h"]

    val_ct = math.floor(brains.shape[0] * opt["split"]["val"])
    test_ct = math.ceil(brains.shape[0] * opt["split"]["test"])
    train_ct = brains.shape[0] - val_ct - test_ct
    pretest_ct = train_ct + val_ct

    # Create the train_x and train_y
    train_x = np.empty(shape=(train_ct, w, h))
    train_y = np.empty(shape=train_ct)

    # Create the val_x and val_y
    val_x = np.empty(shape=(val_ct, w, h))
    val_y = np.empty(shape=val_ct)

    # Create the test_x and test_y
    test_x = np.empty(shape=(test_ct, w, h))
    test_y = np.empty(shape=test_ct)

    for i in range(train_ct):
        img = brains[i, :, img_slice, :]

        train_x[i] = img
        train_y[i] = side

    a = 0
    for j in range(train_ct, pretest_ct):
        img = brains[j, :, img_slice, :]

        val_x[a] = img
        val_y[a] = side
        a += 1

    b = 0
    for k in range(pretest_ct, brains.shape[0]):
        img = brains[k, :, img_slice, :]

        test_x[b] = img
        test_y[b] = side
        b += 1

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_dataset(brains_l, brains_r, opt):

    image_count = opt["scan"]["image_count"]
    use_hipp = opt["scan"]["use_hipp_vol"]
    use_icv = opt["scan"]["use_icv_adj_hipp"]
    use_asym = opt["scan"]["use_icv_adj_hipp"]

    channel_vect = np.arange(image_count)
    s = image_count
    if use_hipp:
        channel_vect = np.append(channel_vect, s)
        channel_vect = np.append(channel_vect, s+1)
        s += 2

    if use_icv:
        channel_vect = np.append(channel_vect, s)
        channel_vect = np.append(channel_vect, s+1)
        s += 2

    if use_asym:
        channel_vect = np.append(channel_vect, s)
        s += 1

    train_x, val_x, test_x = list(), list(), list()
    train_y_brain, val_y_brain, test_y_brain = list(), list(), list()

    for i in channel_vect:
        count = 0
        train_x_l, train_y_l, val_x_l, val_yl, test_x_l, test_y_l = load_img(brains_l, 0, opt, i)
        train_x_r, train_y_r, val_x_r, val_y_r, test_x_r, test_y_r = load_img(brains_r, 1, opt, i)

        train_x_brain = np.concatenate((train_x_l, train_x_r))
        val_x_brain = np.concatenate((val_x_l, val_x_r))
        test_x_brain = np.concatenate((test_x_l, test_x_r))
        train_x.append(train_x_brain)
        val_x.append(val_x_brain)
        test_x.append(test_x_brain)

        if count == 0:

            train_y_brain = to_categorical(np.concatenate((train_y_l, train_y_r)))
            val_y_brain = to_categorical(np.concatenate((val_yl, val_y_r)))
            test_y_brain = to_categorical(np.concatenate((test_y_l, test_y_r)))
            count += 1

    train_x_final = np.stack(train_x, axis=3)
    val_x_final = np.stack(val_x, axis=3)
    test_x_final = np.stack(test_x, axis=3)

    return train_x_final, train_y_brain, val_x_final, val_y_brain, test_x_final, test_y_brain


def load_data(data, side, opt):

    val_ct = math.floor(data.shape[0] * opt["split"]["val"])
    test_ct = math.ceil(data.shape[0] * opt["split"]["test"])
    train_ct = data.shape[0] - val_ct - test_ct
    pretest_ct = train_ct + val_ct

    # Create the train_x and train_y
    train_x = np.empty(shape=(train_ct, 2))
    train_y = np.empty(shape=train_ct)

    # Create the val_x and val_y
    val_x = np.empty(shape=(val_ct, 2))
    val_y = np.empty(shape=val_ct)

    # Create the test_x and test_y
    test_x = np.empty(shape=(test_ct, 2))
    test_y = np.empty(shape=test_ct)

    for i in range(train_ct):
        train_x[i] = data[i, :]
        train_y[i] = side

    a = 0
    for j in range(train_ct, pretest_ct):
        val_x[a] = data[j, :]
        val_y[a] = side
        a += 1

    b = 0
    for k in range(pretest_ct, data.shape[0]):
        test_x[b] = data[k, :]
        test_y[b] = side
        b += 1

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_log_dataset(data_l, data_r, opt):

    train_x_l, train_y_l, val_x_l, val_y_l, test_x_l, test_y_l = load_data(data_l, 0, opt)
    train_x_r, train_y_r, val_x_r, val_y_r, test_x_r, test_y_r = load_data(data_r, 1, opt)

    train_x_data = np.concatenate((train_x_l, train_x_r))
    val_x_data = np.concatenate((val_x_l, val_x_r))
    test_x_data = np.concatenate((test_x_l, test_x_r))

    train_y_data = np.concatenate((train_y_l, train_y_r))
    val_y_data = np.concatenate((val_y_l, val_y_r))
    test_y_data = np.concatenate((test_y_l, test_y_r))

    return train_x_data, train_y_data, val_x_data, val_y_data, test_x_data, test_y_data


def summarize_diagnostics(histories, exp, opt):

    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')

        # plot accuary
        plt.subplot(2, 1, 2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')

    hist_filepath = '../figures/' + exp + '/histories.png'
    plt.tight_layout()
    plt.savefig(hist_filepath)
    plt.show()
