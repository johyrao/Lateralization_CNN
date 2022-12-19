import argparse
import math
from functions import load_opt, load_brains, load_dataset, plot_model_stats, plot_training_stats
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train(exp_name, slice_num, img_data, vol_data, opt):
    channels = len(channel_vect)
    thresholds = np.arange(100) / 100

    scores_val = np.zeros(shape=(runs, thresholds.shape[0]))
    scores_test = np.zeros(shape=(runs, thresholds.shape[0]))
    #     scores_rand = np.zeros(shape=(runs,thresholds.shape[0]))
    #     scores_log = np.zeros(shape=(runs))

    histories = list()

    mean_fpr = np.linspace(0, 1, 100)
    fpr_mat, tpr_mat, auc_mat = list(), list(), list()
    #     fpr_mat_rand, tpr_mat_rand, auc_mat_rand = list(), list(), list()
    #     fpr_mat_log, tpr_mat_log, auc_mat_log = list(), list(), list()

    confusion_mat = np.zeros(shape=(runs, 4))
    #     confusion_mat_rand = np.zeros(shape=(runs,4))
    #     confusion_mat_log = np.zeros(shape=(runs,4))

    scoreboard_left = np.zeros(shape=(brainR.shape[0], runs))
    scoreboard_right = np.zeros(shape=(brainR.shape[0], runs))
    scoreboard_left[scoreboard_left == 0] = np.nan
    scoreboard_right[scoreboard_right == 0] = np.nan

    Lval_ct = math.floor(brainR.shape[0] * .15)
    Ltest_ct = math.ceil(brainR.shape[0] * .15)
    Ltrain_ct = brainR.shape[0] - Lval_ct - Ltest_ct

    Rval_ct = math.floor(brainR.shape[0] * .15)
    Rtest_ct = math.ceil(brainR.shape[0] * .15)
    Rtrain_ct = brainR.shape[0] - Rval_ct - Rtest_ct

    experiment = exp_name

    for i in range(runs):

        print("Run %d of %d" % (i + 1, runs))
        listL = np.random.permutation(brainL.shape[0])
        listR = np.random.permutation(brainR.shape[0])
        brainL_shuffle = brainL[listL][:160]
        brainR_shuffle = brainR[listR]
        volL_shuffle = volL[listL][:160]
        volR_shuffle = volR[listR]
        Ltest_ind = listL[-Ltest_ct:][:160]
        Rtest_ind = listR[-Rtest_ct:]
        test_ind = np.concatenate((Ltest_ind, Rtest_ind), axis=0)
        acc_val, acc_test, acc_rand = list(), list(), list()

        # load dataset
        trainX, trainY, valX, valY, testX, testY = load_dataset(brainL_shuffle, brainR_shuffle, slice_num,
                                                                          channel_vect)
        #         trainX_log, trainY_log, valX_log, valY_log, testX_log, testY_log = load_dataset_log(volL_shuffle,
        #         volR_shuffle)
        #         print(listL)
        #         print(Ltest_ind)
        #         print(listR)
        #         print(Rtest_ind)
        #         print(test_ind)
        print(trainX.shape, valX.shape, testX.shape)
        #         print(trainY.shape, valY.shape, testY.shape)
        #         print(trainX_log.shape, valX_log.shape, testX_log.shape)
        #         print(trainY_log.shape, valY_log.shape, testY_log.shape)
        print("Done loading slice")

        #         subject_performance.append(test_name[:,0])

        #         # Randomly shuffle the y labels
        #         rand_train = np.random.permutation(trainY.shape[0])
        #         rand_val = np.random.permutation(valY.shape[0])
        #         rand_trainY = trainY[rand_train]
        #         rand_valY = valY[rand_val]

        # define model
        cnn_model = define_model_new(channels)
        #         rand_model = define_model(channels)

        # define early stop to stop after validation loss stop going down
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15)
        #         rand_es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=5)

        # define model checkpoint save best performing model
        checkpoint_filepath = "../models/" + experiment + "/slice_" + str(slice_num + 84) + "_" + str(
            channels) + "_channels_run_" + str(i) + ".h5"
        #         rand_checkpoint_filepath = "../models/"+experiment+"/rand_slice_"+str(slice_num+84)+"_"+str(channels)
        #         +"_channels_run_"+str(i)+".h5"
        mc = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_accuracy', mode='max', verbose=0,
                             save_best_only=True)
        #         rand_mc = ModelCheckpoint(filepath=rand_checkpoint_filepath, monitor='val_accuracy', mode='max',
        #         verbose=0, save_best_only=True)

        # fit model
        history = cnn_model.fit(trainX, trainY, epochs=60, batch_size=4, validation_data=(valX, valY), verbose=0,
                                callbacks=[es, mc])
        #         rand_history = rand_model.fit(trainX, rand_trainY, epochs=60, batch_size=4, validation_data=(valX,
        #         rand_valY), verbose=0, callbacks=[rand_es, rand_mc])

        # Evaluate on test
        best_model = load_model(checkpoint_filepath)
        #         rand_best_model = load_model(rand_checkpoint_filepath)
        predY_val = best_model.predict(valX)
        predY_test = best_model.predict(testX)
        #         rand_predY_test = rand_best_model.predict(testX)

        for thresh in thresholds:
            # For val
            predY_val_binary = np.zeros(predY_val[:, 0].shape)
            predY_val_binary[predY_val[:, 0] > thresh] = 1

            acc_val.append(np.sum(valY[:, 0] == predY_val_binary) / valY.shape[0])

            # For test
            predY_test_binary = np.zeros(predY_test[:, 0].shape)
            predY_test_binary[predY_test[:, 0] > thresh] = 1

            acc_test.append(np.sum(testY[:, 0] == predY_test_binary) / testY.shape[0])

        #             #For noting subject by subject accuracies
        #             if thresh == .5:
        #                 for test_sbj_ind in range(test_ind.shape[0]):
        #                     if test_sbj_ind < Ltest_ind.shape[0]:
        #                         if testY[test_sbj_ind,0] == predY_test_binary[test_sbj_ind]:
        #                             scoreboard_left[test_ind[test_sbj_ind],i] = 1
        #                         else:
        #                             scoreboard_left[test_ind[test_sbj_ind],i] = 0
        #                     else:
        #                         if testY[test_sbj_ind,0] == predY_test_binary[test_sbj_ind]:
        #                             scoreboard_right[test_ind[test_sbj_ind],i] = 1
        #                         else:
        #                             scoreboard_right[test_ind[test_sbj_ind],i] = 0

        #             #For rand
        #             rand_predY_test_binary = np.zeros(rand_predY_test[:,0].shape)
        #             rand_predY_test_binary[rand_predY_test[:,0] > thresh] = 1

        #             acc_rand.append(np.sum(testY[:,0] == rand_predY_test_binary) / testY.shape[0])

        print('validation accuarcy: %.3f' % (np.max(acc_val) * 100.0))
        print('Test accuarcy (val): %.3f' % (acc_test[acc_val.index(np.max(acc_val))] * 100.0))
        print('Test accuarcy (0.5): %.3f' % (acc_test[49] * 100.0))
        #         print('Test accuarcy (max): %.3f' % (np.max(acc_test) * 100.0))
        #         print('Test accuarcy (rand): %.3f' % (acc_rand[49] * 100.0))

        # stores scores
        scores_val[i, :] = acc_val
        scores_test[i, :] = acc_test
        #         scores_rand[i,:] = acc_rand
        histories.append(history)

        # evaluate model on test dataset to get auc
        fpr, tpr, treshhold_roc = roc_curve(testY[:, 0], predY_test[:, 0], drop_intermediate=False)
        #         fpr_rand, tpr_rand, treshhold_roc_rand = roc_curve(testY[:,0], rand_predY_test[:,0])
        auc_cnn = auc(fpr, tpr)
        #         auc_rand = auc(fpr_rand, tpr_rand)

        tpr_mat.append(np.interp(mean_fpr, fpr, tpr))
        #         tpr_mat_rand.append(np.interp(mean_fpr, fpr_rand, tpr_rand))

        # Produce the confusion matrix
        pred = np.argmax(predY_test, axis=1)
        #         pred_rand = np.argmax(rand_predY_test, axis=1)
        label = np.argmax(testY, axis=1)
        con_mat = confusion_matrix(label, pred)
        #         con_mat_rand = confusion_matrix(label, pred_rand)

        auc_mat.append(auc_cnn)
        #         auc_mat_rand.append(auc_rand)
        confusion_mat[i, 0] = con_mat[0, 0]
        confusion_mat[i, 1] = con_mat[0, 1]
        confusion_mat[i, 2] = con_mat[1, 0]
        confusion_mat[i, 3] = con_mat[1, 1]
    #         confusion_mat_rand[i,0]=con_mat_rand[0,0]
    #         confusion_mat_rand[i,1]=con_mat_rand[0,1]
    #         confusion_mat_rand[i,2]=con_mat_rand[1,0]
    #         confusion_mat_rand[i,3]=con_mat_rand[1,1]

    #         ## SVM Section
    #         log_model = linear_model.LogisticRegression(random_state=0).fit(trainX_log, trainY_log)
    #         predY_test_log = log_model.decision_function(testX_log)
    #         predY_binary_log = log_model.predict(testX_log)

    #         fpr_log, tpr_log, threshold_roc_log = roc_curve(testY_log, predY_test_log)
    #         auc_log = auc(fpr_log, tpr_log)
    #         con_mat_log = confusion_matrix(testY_log, predY_binary_log)

    #         tpr_mat_log.append(np.interp(mean_fpr, fpr_log, tpr_log))

    #         auc_mat_log.append(auc_log)
    #         confusion_mat_log[i,0]=con_mat_log[0,0]
    #         confusion_mat_log[i,1]=con_mat_log[0,1]
    #         confusion_mat_log[i,2]=con_mat_log[1,0]
    #         confusion_mat_log[i,3]=con_mat_log[1,1]

    #         acc_log = log_model.score(testX_log, testY_log)
    #         scores_log[i] = acc_log
    #         print('Test accuarcy (log): %.3f' % (acc_log * 100.0))

    #         plt.figure(1)
    #         plt.plot([0, 1], [0, 1], 'k--')
    #         plt.plot(fpr, tpr, label='CNN (area = {:.3f})'.format(auc_cnn))
    #         plt.plot(fpr_log, tpr_log, label='Log Regression (area = {:.3f})'.format(auc_log))
    #         plt.plot(fpr_rand, tpr_rand, label='CNN Random (area = {:.3f})'.format(auc_rand))
    #         plt.xlabel('False positive rate')
    #         plt.ylabel('True positive rate')
    #         plt.title('ROC curve')
    #         plt.legend(loc='best')
    #         roc_filepath = '../figures/'+experiment+'/roc_curve_run_'+str(i)+'.png'
    #         plt.savefig(roc_filepath)
    #         plt.clf()

    # Write score_val
    filename_scores_val = '../files/' + experiment + '_scores_val.csv'
    np.savetxt(filename_scores_val, scores_val, delimiter=',', fmt='%1.3f')

    # Write score_test
    filename_scores_test = '../files/' + experiment + '_scores_test.csv'
    np.savetxt(filename_scores_test, scores_test, delimiter=',', fmt='%1.3f')

    # Write TPR
    filename_tpr = '../files/' + experiment + '_tpr.csv'
    np.savetxt(filename_tpr, tpr_mat, delimiter=',', fmt='%1.3f')

    # Write AUC
    filename_auc = '../files/' + experiment + '_auc.csv'
    np.savetxt(filename_auc, auc_mat, delimiter=',', fmt='%1.3f')

    # Write Confusion matrix
    filename_con_mat = '../files/' + experiment + '_con_mat.csv'
    np.savetxt(filename_con_mat, confusion_mat, delimiter=',', fmt='%1.3f')

    #     # Write score_test Rand
    #     filename_score_test_rand = '../files/'+experiment+'_scores_test_rand.csv'
    #     np.savetxt(filename_score_test_rand,scores_rand,delimiter=',')

    #     # Write TPR
    #     filename_tpr_rand = '../files/'+experiment+'_tpr_rand.csv'
    #     np.savetxt(filename_tpr_rand,tpr_mat_rand,delimiter=',',fmt='%1.3f')

    #     # Write AUC Rand
    #     filename_auc_rand = '../files/'+experiment+'_auc_rand.csv'
    #     np.savetxt(filename_auc_rand,auc_mat_rand,delimiter=',')

    #     # Write Confusion matrix Rand
    #     filename_con_mat_rand = '../files/'+experiment+'_con_mat_rand.csv'
    #     np.savetxt(filename_con_mat_rand,confusion_mat_rand,delimiter=',')

    #     # Write score_test Log
    #     filename_scores_log = '../files/'+experiment+'_scores_log.csv'
    #     np.savetxt(filename_scores_log,scores_log,delimiter=',')

    #     # Write TPR
    #     filename_tpr_log = '../files/'+experiment+'_tpr_log.csv'
    #     np.savetxt(filename_tpr_log,tpr_mat_log,delimiter=',',fmt='%1.3f')

    #     # Write AUC Log
    #     filename_auc_log = '../files/'+experiment+'_auc_log.csv'
    #     np.savetxt(filename_auc_log,auc_mat_log,delimiter=',')

    #     # Write Confusion matrix Log
    #     filename_con_mat_log = '../files/'+experiment+'_con_mat_log.csv'
    #     np.savetxt(filename_con_mat_log,confusion_mat_log,delimiter=',')

    #     # Write sbj scores Left TLE
    #     filename_scores_left_sbj = '../files/'+experiment+'_scores_left_sbj.csv'
    #     np.savetxt(filename_scores_left_sbj,scoreboard_left,delimiter=',',fmt='%1.3f')

    #     # Write sbj scores Right TLE
    #     filename_scores_right_sbj = '../files/'+experiment+'_scores_right_sbj.csv'
    #     np.savetxt(filename_scores_right_sbj,scoreboard_right,delimiter=',',fmt='%1.3f')

    print(img_data, vol_data, opt)
    return exp_name, slice_num


def main():
    print("hello world")
    parser = argparse.ArgumentParser(description="Run CNN", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("exp", help="Experiment name")
    parser.add_argument("slice", help="Which coronal slice to start input from")

    args = vars(parser.parse_args())
    exp_name = args["exp"]
    slice_num = args["slice"]

    opt = load_opt("D:\\Lab\\Lateralization_CNN\\setting\\")

    # Store all the different groups as separate index
    img_data = list()
    vol_data = list()

    for grp in opt["group"]["group_list"]:
        img, vol = load_brains(grp, opt)
        img_data.append(img)
        vol_data.append(vol)

    results, histories = train(exp_name, slice_num, img_data, vol_data, opt)
    print("Done training models")

    plot_training_stats(histories, exp_name, opt)
    plot_model_stats(results, exp_name, opt)


if __name__ == "__main__":
    main()
