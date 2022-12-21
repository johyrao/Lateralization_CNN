import cv2
import json
import math
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


def normalization(array):
    norm_arr = (array - np.nanmin(array))/(np.nanmax(array) - np.nanmin(array))
    return norm_arr


def load_opt(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def load_brains(side, opt):
    dir_list = opt["sites"]["site_list"]
    image_path = opt["filepath"]["images"]
    vol_path = opt["filepath"]["volumes"]
    image_count = opt["scan"]["image_count"]
    extra_count = opt["scan"]["extra_count"]
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

    total_slices = image_count + extra_count

    filenames = np.empty(shape=(0,), dtype=str)
    brains_comb = np.empty(shape=(0, w, total_slices, h))
    vol_vect = np.empty(shape=(0, 2))

    for d in dir_list:
        dir_path_full = image_path + side + "\\" + d
        print(dir_path_full)

        for (root, dirs, files) in os.walk(dir_path_full):
            num_files = len(files)  # get number of files in the dir
            brains = np.zeros(shape=(num_files, w, total_slices, h))  # saves all the brains
            vols = np.zeros(shape=(num_files, 2))
            filename = list()

            for i in range(num_files):
                img = nib.load(dir_path_full + "\\" + files[i]).get_fdata()  # load the subject brain
                brains[i, :, :image_count, :] = img[l_ind:r_ind, p_ind:a_ind:offset, i_ind:s_ind]
                filename.append(files[i])

        brains[brains == 0] = np.nan
        norm_brains = [normalization(i) for i in brains]
        norm_brains = np.array(norm_brains)
        norm_brains[np.isnan(norm_brains)] = 0

        if extra_count > 0:
            # Load volumes
            file_path_vol = vol_path + side + "\\" + 'icv_vol_' + d + '.csv'
            print(file_path_vol)

            df_vol = pd.read_csv(file_path_vol, header=None)
            hipp_l = df_vol.iloc[:, 0]
            hipp_r = df_vol.iloc[:, 1]
            icv = df_vol.iloc[:, 2]
            asym = df_vol.iloc[:, 3]

            for i in range(num_files):
                slice1 = np.full((w, h), hipp_l[i])
                slice2 = np.full((w, h), hipp_r[i])
                s = image_count

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

        filenames = np.concatenate((filenames, filename))
        brains_comb = np.concatenate((brains_comb, norm_brains))
        vol_vect = np.concatenate((vol_vect, vols))

    return filenames, brains_comb, vol_vect


def load_data(img_data, vol_data, names, opt):
    for grp in opt["group"]["group_list"]:
        filenames, img, vol = load_brains(grp, opt)
        names.append(filenames)
        img_data.append(img)
        vol_data.append(vol)

    return names, img_data, vol_data


def load_img(images, side, opt, img_slice):
    w = opt["scan"]["size_w"]
    h = opt["scan"]["size_h"]

    val_ct = math.floor(images.shape[0] * opt["split"]["val"])
    test_ct = math.ceil(images.shape[0] * opt["split"]["test"])
    train_ct = images.shape[0] - val_ct - test_ct
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
        img = images[i, :, img_slice, :]

        train_x[i] = img
        train_y[i] = side

    for count, j in enumerate(range(train_ct, pretest_ct)):
        img = images[j, :, img_slice, :]

        val_x[count] = img
        val_y[count] = side

    for count, k in enumerate(range(pretest_ct, images.shape[0])):
        img = images[k, :, img_slice, :]

        test_x[count] = img
        test_y[count] = side

    return train_x, train_y, val_x, val_y, test_x, test_y


def load_dataset(data, opt):
    image_count = opt["scan"]["image_count"]
    use_hipp = opt["scan"]["use_hipp_vol"]
    use_icv = opt["scan"]["use_icv_adj_hipp"]
    use_asym = opt["scan"]["use_icv_adj_hipp"]
    w = opt["scan"]["size_w"]
    h = opt["scan"]["size_h"]

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
    train_y_img = np.empty(shape=0)
    val_y_img = np.empty(shape=0)
    test_y_img = np.empty(shape=0)
    count = 0

    for i in channel_vect:
        train_x_img = np.empty(shape=(0, w, h))
        val_x_img = np.empty(shape=(0, w, h))
        test_x_img = np.empty(shape=(0, w, h))

        for grp_num, grp_data in enumerate(data):
            train_x_grp, train_y_grp, val_x_grp, val_y_grp, test_x_grp, test_y_grp = load_img(grp_data, grp_num, opt, i)

            train_x_img = np.concatenate((train_x_img, train_x_grp))
            val_x_img = np.concatenate((val_x_img, val_x_grp))
            test_x_img = np.concatenate((test_x_img, test_x_grp))

            if count == 0:
                train_y_img = np.concatenate((train_y_img, train_y_grp))
                val_y_img = np.concatenate((val_y_img, val_y_grp))
                test_y_img = np.concatenate((test_y_img, test_y_grp))

        train_x.append(train_x_img)
        val_x.append(val_x_img)
        test_x.append(test_x_img)
        count += 1

    train_x_final = np.stack(train_x, axis=3)
    val_x_final = np.stack(val_x, axis=3)
    test_x_final = np.stack(test_x, axis=3)
    train_y = to_categorical(train_y_img)
    val_y = to_categorical(val_y_img)
    test_y = to_categorical(test_y_img)

    return train_x_final, train_y, val_x_final, val_y, test_x_final, test_y


def load_value(data, side, opt):
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


def load_regression_dataset(data, opt):

    train_x_data = np.empty(shape=(0, 2))
    val_x_data = np.empty(shape=(0, 2))
    test_x_data = np.empty(shape=(0, 2))
    train_y_data = np.empty(shape=0)
    val_y_data = np.empty(shape=0)
    test_y_data = np.empty(shape=0)

    for grp_num, grp_data in enumerate(data):
        train_x_grp, train_y_grp, val_x_grp, val_y_grp, test_x_grp, test_y_grp = load_value(grp_data, grp_num, opt)

        train_x_data = np.concatenate((train_x_data, train_x_grp))
        val_x_data = np.concatenate((val_x_data, val_x_grp))
        test_x_data = np.concatenate((test_x_data, test_x_grp))

        train_y_data = np.concatenate((train_y_data, train_y_grp))
        val_y_data = np.concatenate((val_y_data, val_y_grp))
        test_y_data = np.concatenate((test_y_data, test_y_grp))

    return train_x_data, train_y_data, val_x_data, val_y_data, test_x_data, test_y_data


def plot_training_stats(histories, exp, opt):
    fig_path = opt["filepath"]["figures"]

    for i in range(len(histories)):
        # plot loss
        plt.subplot(2, 1, 1)
        plt.title("Cross Entropy Loss")
        plt.plot(histories[i].history["loss"], color="blue", label="train")
        plt.plot(histories[i].history["val_loss"], color="orange", label="test")

        # plot accuracy
        plt.subplot(2, 1, 2)
        plt.title("Classification Accuracy")
        plt.plot(histories[i].history["accuracy"], color="blue", label="train")
        plt.plot(histories[i].history["val_accuracy"], color="orange", label="test")

    hist_filepath = fig_path + exp + "\\histories.png"
    plt.tight_layout()
    plt.savefig(hist_filepath)


def plot_model_stats(results, exp, opt):
    fig_path = opt["filepath"]["figures"]
    run_rand = opt["model"]["random_CNN"]
    run_log = opt["model"]["logistic_regression"]
    val = results["val"]
    test = results["test"]

    # print summary Val
    val_max = np.max(val, axis=1)
    print("Val Accuracy (max): mean=%.3f, std=%.3f, n=%d" % (np.mean(val_max) * 100, np.std(val_max) * 100,
                                                             val.shape[0]))

    # print summary Test (val)
    test_val = np.zeros(shape=(val.shape[0]))
    for i in range(val.shape[0]):
        ind = np.where(val[i, :] == np.max(val[i, :]))
        test_val[i] = test[i, ind[0][0]]
    print("Test Accuracy (val): mean=%.3f, std=%.3f, n=%d" % (np.mean(test_val) * 100, np.std(test_val) * 100,
                                                              test.shape[0]))

    # print summary Test (50)
    test_50 = np.zeros(shape=(test.shape[0]))
    for i in range(test.shape[0]):
        test_50[i] = test[i, 49]
    print("Test Accuracy (50): mean=%.3f, std=%.3f, n=%d" % (np.mean(test_50) * 100, np.std(test_50) * 100,
                                                             test.shape[0]))

    data = {"CNN (val)": test_val, "CNN (50)": test_50}

    if run_rand:
        # print summary Random Model
        rand = results["random_CNN"]
        rand_50 = np.zeros(shape=(rand.shape[0]))

        for i in range(rand.shape[0]):
            rand_50[i] = rand[i, 49]
        print("Test Accuracy (rand): mean=%.3f, std=%.3f, n=%d" % (np.mean(rand_50)*100, np.std(rand_50)*100,
                                                                   rand.shape[0]))

        data["Random Model"] = rand_50

    if run_log:
        # print summary Logistic Regression
        log = results["logistic_regression"]

        print("Test Accuracy (log): mean=%.3f, std=%.3f, n=%d" % (np.mean(log)*100, np.std(log)*100, log.shape[0]))

        data["Logistic Regression"] = log

    fig, ax = plt.subplots()
    acc_filepath = fig_path + exp + "\\acc_boxplot.png"
    ax.boxplot(data.values())
    plt.title("Accuracies of Different Models")
    ax.set_xticklabels(data.keys())
    plt.savefig(acc_filepath)


def save_csv_file(data, exp_name, file, opt):
    filename = opt["filepath"]["files"] + exp_name + "\\" + file + ".csv"
    np.savetxt(filename, data, delimiter=',', fmt='%1.3f')


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, pred = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(pred[0])
        class_channel = pred[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def get_hm_by_diff_sbj(model_name, img_data, opt):
    channels = opt["scan"]["image_count"] + opt["scan"]["extra_count"]

    # run_model(slice_start, channels, brainsL, brainsR)
    model_cnn = load_model(model_name)

    heatmaps = list()
    for grp, data in enumerate(img_data):
        heatmap_mat = np.empty(shape=(40, 40, data.shape[0]))

        # Get third convolutional layer name
        conv_index = 0
        layer_name = ""
        for layer in model_cnn.layers:

            if 'conv' in layer.name:
                conv_index = conv_index + 1
                if conv_index == 3:
                    layer_name = layer.name

        for i in range(data.shape[0]):
            test_x = np.empty(shape=(1, opt["scan"]["size_w"], opt["scan"]["size_h"], channels))
            for j in range(channels):
                img = data[i, :, j, :]
                test_x[0, :, :, j] = img

            heatmap = make_gradcam_heatmap(test_x, model_cnn, layer_name)
            heatmap_mat[:, :, i] = heatmap

        heatmaps.append(heatmap_mat)

    return heatmaps


def run_heatmap(img_data, slice_num, runs, exp_name, opt):
    channels = opt["scan"]["image_count"] + opt["scan"]["extra_count"]

    heatmap = list()
    for data in img_data:
        heatmap.append(np.empty(shape=(40, 40, data.shape[0], runs)))

    for i in range(runs):
        # Create model filename string
        print('Getting model heatmaps for %d of %d model runs' % (i + 1, runs))
        model_name = opt["filepath"]["models"] + exp_name + "/slice_" + str(slice_num) + "_" + str(
            channels) + "_channels_run_" + str(i) + ".h5"

        heatmaps = get_hm_by_diff_sbj(model_name, img_data, opt)

        for grp, hm in enumerate(heatmap):
            hm[:, :, :, i] = heatmaps[grp]

    for grp, hm in enumerate(heatmap):
        heatmap_avg_model = np.mean(hm, axis=3)
        heatmap_avg_sbj = np.mean(heatmap_avg_model, axis=2)
        heatmap_resize = cv2.resize(heatmap_avg_sbj, (opt["scan"]["size_w"], opt["scan"]["size_h"]),
                                    interpolation=cv2.INTER_CUBIC)

        plt.imshow(np.rot90(img_data[grp][1, :, 4, :]), cmap="gray")
        plt.imshow(np.rot90(heatmap_resize), cmap="jet", alpha=0.4)
        heatmap_path = opt["filepath"]["figures"] + exp_name + "\\heatmap_" + str(grp) + ".png"
        plt.savefig(heatmap_path)
