import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from functions import load_opt, load_data, load_dataset, load_regression_dataset, plot_model_stats, \
    plot_training_stats, save_csv_file, run_heatmap
from make_models import define_model, define_model_3d
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model


def train(exp_name, slice_num, img_data, vol_data, opt):
    runs = opt["training"]["runs"]
    run_rand = opt["model"]["random_CNN"]
    run_log = opt["model"]["logistic_regression"]

    channels = opt["scan"]["image_count"] + opt["scan"]["extra_count"]
    cnn_3d = opt["model"]["3D"]
    thresholds = np.arange(100) / 100

    scores_val = np.zeros(shape=(runs, thresholds.shape[0]))
    scores_test = np.zeros(shape=(runs, thresholds.shape[0]))
    histories = list()
    mean_fpr = np.linspace(0, 1, 100)
    fpr_mat, tpr_mat, auc_mat = list(), list(), list()
    confusion_mat = np.zeros(shape=(runs, 4))

    if run_rand:
        scores_rand = np.zeros(shape=(runs, thresholds.shape[0]))
        fpr_mat_rand, tpr_mat_rand, auc_mat_rand = list(), list(), list()
        confusion_mat_rand = np.zeros(shape=(runs, 4))

    if run_log:
        scores_log = np.zeros(shape=runs)
        fpr_mat_log, tpr_mat_log, auc_mat_log = list(), list(), list()
        confusion_mat_log = np.zeros(shape=(runs, 4))

    scoreboards, val_ct_classes, test_ct_classes, train_ct_classes = list(), list(), list(), list()
    for grp, data in enumerate(img_data):
        scoreboard = np.zeros(shape=(data.shape[0], runs))
        scoreboard[scoreboard == 0] = np.nan
        scoreboards.append(scoreboard)

        val_ct = math.floor(data.shape[0] * opt["split"]["val"])
        val_ct_classes.append(val_ct)

        test_ct = math.ceil(data.shape[0] * opt["split"]["test"])
        test_ct_classes.append(test_ct)

        train_ct = data.shape[0] - val_ct - test_ct
        train_ct_classes.append(train_ct)

    for i in range(runs):
        print("Run %d of %d" % (i + 1, runs))
        acc_val, acc_test, acc_rand = list(), list(), list()
        data_shuffle_classes, vol_shuffle_classes, test_ind_classes = list(), list(), list()

        for grp, data in enumerate(img_data):
            list_grp = np.random.permutation(data.shape[0])
            data_shuffle = data[list_grp]
            data_shuffle_classes.append(data_shuffle)

            vol_shuffle = vol_data[grp][list_grp]
            vol_shuffle_classes.append(vol_shuffle)

            test_ind = list_grp[-test_ct_classes[grp]:]
            test_ind_classes.append(test_ind)

        # load dataset
        train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(data_shuffle_classes, opt)
        print(train_x.shape, val_x.shape, test_x.shape)
        print("Done loading slice")

        # define model
        if cnn_3d:
            cnn_model = define_model_3d(channels, opt)
        else:
            cnn_model = define_model(channels, opt)

        # define early stop to stop after validation loss stop going down
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
        # define model checkpoint save best performing model
        checkpoint_filepath = opt["filepath"]["models"] + exp_name + "/slice_" + str(slice_num) + "_" + str(
            channels) + "_channels_run_" + str(i) + ".h5"
        mc = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', verbose=0,
                             save_best_only=True)

        # fit model
        history = cnn_model.fit(train_x, train_y, epochs=60, batch_size=4, validation_data=(val_x, val_y), verbose=0,
                                callbacks=[es, mc])

        # Evaluate on test
        best_model = load_model(checkpoint_filepath)
        pred_y_val = best_model.predict(val_x)
        pred_y_test = best_model.predict(test_x)

        for thresh in thresholds:
            # For val
            pred_y_val_binary = np.zeros(pred_y_val[:, 0].shape)
            pred_y_val_binary[pred_y_val[:, 0] > thresh] = 1

            acc_val.append(np.sum(val_y[:, 0] == pred_y_val_binary) / val_y.shape[0])

            # For test
            pred_y_test_binary = np.zeros(pred_y_test[:, 0].shape)
            pred_y_test_binary[pred_y_test[:, 0] > thresh] = 1

            acc_test.append(np.sum(test_y[:, 0] == pred_y_test_binary) / test_y.shape[0])

            # For noting subject by subject accuracies
            if thresh == .5:
                for grp, test_ind in enumerate(test_ind_classes):
                    for test_sbj_ind in range(test_ind.shape[0]):
                        if test_y[test_sbj_ind, 0] == pred_y_test_binary[test_sbj_ind]:
                            scoreboards[grp][test_sbj_ind, i] = 1
                        else:
                            scoreboards[grp][test_sbj_ind, i] = 0

        print('validation accuracy: %.3f' % (np.max(acc_val) * 100.0))
        print('Test accuracy (val): %.3f' % (acc_test[acc_val.index(np.max(acc_val))] * 100.0))
        print('Test accuracy (0.5): %.3f' % (acc_test[49] * 100.0))

        # Store scores
        scores_val[i, :] = acc_val
        scores_test[i, :] = acc_test
        histories.append(history)

        # evaluate model on test dataset to get auc
        fpr, tpr, threshold_roc = roc_curve(test_y[:, 0], pred_y_test[:, 0], drop_intermediate=False)
        tpr_mat.append(np.interp(mean_fpr, fpr, tpr))
        auc_cnn = auc(fpr, tpr)
        auc_mat.append(auc_cnn)

        # Produce the confusion matrix
        pred = np.argmax(pred_y_test, axis=1)
        label = np.argmax(test_y, axis=1)
        con_mat = confusion_matrix(label, pred)
        confusion_mat[i, 0] = con_mat[0, 0]
        confusion_mat[i, 1] = con_mat[0, 1]
        confusion_mat[i, 2] = con_mat[1, 0]
        confusion_mat[i, 3] = con_mat[1, 1]

        if run_rand:
            # Randomly shuffle the y labels
            rand_train = np.random.permutation(train_y.shape[0])
            rand_val = np.random.permutation(val_y.shape[0])
            rand_train_y = train_y[rand_train]
            rand_val_y = val_y[rand_val]

            rand_model = define_model(channels, opt)

            rand_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)
            rand_checkpoint_filepath = opt["filepath"]["models"] + exp_name + "/rand_slice_" + str(
                slice_num) + "_" + str(channels) + "_channels_run_" + str(i) + ".h5"
            rand_mc = ModelCheckpoint(filepath=rand_checkpoint_filepath, monitor='val_accuracy', mode='max',
                                      verbose=0, save_best_only=True)
            _ = rand_model.fit(train_x, rand_train_y, epochs=60, batch_size=4, validation_data=(val_x, rand_val_y),
                               verbose=0, callbacks=[rand_es, rand_mc])

            rand_best_model = load_model(rand_checkpoint_filepath)
            rand_pred_y_test = rand_best_model.predict(test_x)

            for thresh in thresholds:
                # For rand
                rand_pred_y_test_binary = np.zeros(rand_pred_y_test[:, 0].shape)
                rand_pred_y_test_binary[rand_pred_y_test[:, 0] > thresh] = 1

                acc_rand.append(np.sum(test_y[:, 0] == rand_pred_y_test_binary) / test_y.shape[0])
                print("Test accuracy (rand): %.3f" % (acc_rand[49] * 100.0))

                scores_rand[i, :] = acc_rand

                fpr_rand, tpr_rand, threshold_roc_rand = roc_curve(test_y[:, 0], rand_pred_y_test[:, 0])
                tpr_mat_rand.append(np.interp(mean_fpr, fpr_rand, tpr_rand))
                auc_rand = auc(fpr_rand, tpr_rand)
                auc_mat_rand.append(auc_rand)

                pred_rand = np.argmax(rand_pred_y_test, axis=1)
                con_mat_rand = confusion_matrix(label, pred_rand)
                confusion_mat_rand[i, 0] = con_mat_rand[0, 0]
                confusion_mat_rand[i, 1] = con_mat_rand[0, 1]
                confusion_mat_rand[i, 2] = con_mat_rand[1, 0]
                confusion_mat_rand[i, 3] = con_mat_rand[1, 1]

        if run_log:
            train_x_log, train_y_log, val_x_log, val_y_log, test_x_log, test_y_log = \
                load_regression_dataset(vol_shuffle_classes, opt)

            # SVM Section
            log_model = linear_model.LogisticRegression(random_state=0).fit(train_x_log, train_y_log)
            pred_y_test_log = log_model.decision_function(test_x_log)
            pred_y_binary_log = log_model.predict(test_x_log)

            fpr_log, tpr_log, threshold_roc_log = roc_curve(test_y_log, pred_y_test_log)
            tpr_mat_log.append(np.interp(mean_fpr, fpr_log, tpr_log))
            auc_log = auc(fpr_log, tpr_log)
            auc_mat_log.append(auc_log)

            con_mat_log = confusion_matrix(test_y_log, pred_y_binary_log)
            confusion_mat_log[i, 0] = con_mat_log[0, 0]
            confusion_mat_log[i, 1] = con_mat_log[0, 1]
            confusion_mat_log[i, 2] = con_mat_log[1, 0]
            confusion_mat_log[i, 3] = con_mat_log[1, 1]

            acc_log = log_model.score(test_x_log, test_y_log)
            scores_log[i] = acc_log
            print("Test accuracy (log): %.3f" % (acc_log * 100.0))

            plt.figure(1)
            plt.plot([0, 1], [0, 1], "k--")
            plt.plot(fpr, tpr, label="CNN (area = {:.3f})".format(auc_cnn))
            plt.plot(fpr_log, tpr_log, label="Log Regression (area = {:.3f})".format(auc_log))
            if run_rand:
                plt.plot(fpr_rand, tpr_rand, label="CNN Random (area = {:.3f})".format(auc_rand))
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title("ROC curve")
            plt.legend(loc="best")
            roc_filepath = opt["filepath"]["figures"] + exp_name + "\\roc_curve_run_" + str(i) + ".png"
            plt.savefig(roc_filepath)
            plt.clf()

    # Write score_val
    save_csv_file(scores_val, exp_name, "scores_val", opt)
    # Write score_test
    save_csv_file(scores_test, exp_name, "scores_test", opt)
    # Write TPR
    save_csv_file(tpr_mat, exp_name, "tpr", opt)
    # Write AUC
    save_csv_file(auc_mat, exp_name, "auc", opt)
    # Write Confusion matrix
    save_csv_file(confusion_mat, exp_name, "con_mat", opt)
    # Write sbj scores TLE
    for grp, scoreboard in enumerate(scoreboards):
        save_csv_file(scoreboard, exp_name, "scores_class_" + str(grp), opt)

    results = {"val": scores_val, "test": scores_test}

    if run_rand:
        # Write score_test Rand
        save_csv_file(scores_rand, exp_name, "scores_test_rand", opt)
        # Write TPR Rand
        save_csv_file(tpr_mat_rand, exp_name, "tpr_rand", opt)
        # Write AUC Rand
        save_csv_file(auc_mat_rand, exp_name, "auc_rand", opt)
        # Write Confusion matrix Rand
        save_csv_file(confusion_mat_rand, exp_name, "con_mat_rand", opt)

        results["rand"] = scores_rand

    if run_log:
        # Write score_test Log
        save_csv_file(scores_log, exp_name, "scores_log", opt)
        # Write TPR Log
        save_csv_file(tpr_mat_log, exp_name, "tpr_log", opt)
        # Write AUC Log
        save_csv_file(auc_mat_log, exp_name, "auc_log", opt)
        # Write Confusion matrix Log
        save_csv_file(confusion_mat_log, exp_name, "con_mat_log", opt)

        results["log"] = scores_log

    return results, histories


def main():
    parser = argparse.ArgumentParser(description="Run CNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("exp", help="Experiment name")
    parser.add_argument("slice", help="Which coronal slice to start input from")
    parser.add_argument("-d", "--density-plot", action="store_true", help="Plot density plots")
    parser.add_argument("-m", "--heatmap", action="store_true", help="Plot heatmaps")

    args = vars(parser.parse_args())
    exp_name = args["exp"]
    slice_num = int(args["slice"])
    density_plot = args["density_plot"]
    get_heatmaps = args["heatmap"]

    opt = load_opt("D:\\Lab\\Lateralization_CNN\\setting\\opts.json")
    if os.path.exists(opt["filepath"]["figures"] + exp_name + "\\") or \
            os.path.exists(opt["filepath"]["files"] + exp_name + "\\") or \
            os.path.exists(opt["filepath"]["models"] + exp_name + "\\"):
        print("Experiment name is taken. Rerun with new name")
        exit()
    if not os.path.exists(opt["filepath"]["figures"] + exp_name + "\\"):
        os.mkdir(opt["filepath"]["figures"] + exp_name + "\\")
    if not os.path.exists(opt["filepath"]["files"] + exp_name + "\\"):
        os.mkdir(opt["filepath"]["files"] + exp_name + "\\")
    if not os.path.exists(opt["filepath"]["models"] + exp_name + "\\"):
        os.mkdir(opt["filepath"]["models"] + exp_name + "\\")

    run_rand = opt["model"]["random_CNN"]
    run_log = opt["model"]["logistic_regression"]

    # Store all the different groups as separate index
    names = list()
    img_data = list()
    vol_data = list()

    names, img_data, vol_data = load_data(img_data, vol_data, names, opt)

    results, histories = train(exp_name, slice_num, img_data, vol_data, opt)
    print("Done training models")

    plot_training_stats(histories, exp_name, opt)
    plot_model_stats(results, exp_name, opt)

    # Compare density plot of CNN and mean Rand and mean Log
    sns.displot(results["test"][:, 49] * 100, kind="kde", label="CNN")
    if run_rand:
        plt.axvline(np.mean(results["log"]) * 100, 0, 1, color="green", label="Log")
    if run_log:
        plt.axvline(np.mean(results["rand"][:, 49]) * 100, 0, 1, color="red", label="Rand")
    plt.legend(loc="best")
    dist_filepath = opt["filepath"]["figures"] + exp_name + "\\distribution_plot.png"
    plt.savefig(dist_filepath)

    if density_plot:
        if run_rand:
            # Density plot of Rand and CNN
            sns.displot((results["test"][:, 49] * 100, results["rand"][:, 49] * 100), kind='kde', legend=False)
            rand_cnn_filepath = opt["filepath"]["figures"] + exp_name + "\\rand_CNN_distribution_plot.png"
            plt.savefig(rand_cnn_filepath)

            # Density plot of CNN minus Rand
            scores_diff_rand = results["test"][:, 49] - results["rand"][:, 49]
            sns.displot(scores_diff_rand * 100, kind='kde')
            diff_rand_filepath = opt["filepath"]["figures"] + exp_name + "\\scores_diff_rand_distribution_plot.png"
            plt.savefig(diff_rand_filepath)

        if run_log:
            # Density plot of Log and CNN
            sns.displot((results["test"][:, 49] * 100, results["log"] * 100), kind='kde', legend=False)
            log_cnn_filepath = opt["filepath"]["figures"] + exp_name + "\\log_CNN_distribution_plot.png"
            plt.savefig(log_cnn_filepath)

            # Density plot of CNN minus Log
            scores_diff_log = results["test"][:, 49] - results["log"]
            sns.displot(scores_diff_log * 100, kind='kde')
            diff_log_filepath = opt["filepath"]["figures"] + exp_name + "\\scores_diff_log_distribution_plot.png"
            plt.savefig(diff_log_filepath)

    if run_rand and run_log:
        diff_data = {'CNN<Rand': np.count_nonzero(scores_diff_rand < 0),
                     'CNN==Rand': np.count_nonzero(scores_diff_rand == 0),
                     'CNN>Rand': np.count_nonzero(scores_diff_rand > 0),
                     'CNN<Log': np.count_nonzero(scores_diff_log < 0),
                     'CNN==Log': np.count_nonzero(scores_diff_log == 0),
                     'CNN>Log': np.count_nonzero(scores_diff_log > 0)}
        diff_df = pd.DataFrame(data=diff_data, index=["Count"])
        diff_df_path = opt["filepath"]["files"] + exp_name + "\\count_diff.csv"
        diff_df.to_csv(diff_df_path)

    if get_heatmaps:
        run_heatmap(img_data, slice_num, 5, exp_name, opt)


if __name__ == "__main__":
    main()
