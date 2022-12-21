import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D, \
    Dropout, Conv3D, MaxPool3D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import regularizers, Input
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale


def make_feature_matrix(train_x, val_x, test_x, exp, run):
    train_feat_mat = np.empty(shape=(train_x.shape[0], 51200))
    val_feat_mat = np.empty(shape=(val_x.shape[0], 51200))
    test_feat_mat = np.empty(shape=(test_x.shape[0], 51200))

    model_path = '../output/' + exp + '/models/ensemble_run_' + str(run) + '.h5'
    model = load_model(model_path)

    model_feature = Model(inputs=model.inputs, outputs=model.layers[6].output)

    for i in range(train_x.shape[0]):
        train_feat_mat[i, :] = model_feature.predict(train_x[i:(i + 1), :, :, :])[0]

    for i in range(val_x.shape[0]):
        val_feat_mat[i, :] = model_feature.predict(val_x[i:(i + 1), :, :, :])[0]

    for i in range(test_x.shape[0]):
        test_feat_mat[i, :] = model_feature.predict(test_x[i:(i + 1), :, :, :])[0]

    return train_feat_mat, val_feat_mat, test_feat_mat


def train_pls(train_feat_mat, train_y, component):
    pls = PLSRegression(scale=False, n_components=component)
    pls.fit(scale(train_feat_mat), train_y[:, 0])

    return pls


def make_resnet(config):
    # Make pretrain resnet model
    inputs = Input(shape=(config["scan"]["size_w"], config["scan"]["size_h"], 3))

    res_net = ResNet50V2(include_top=False, input_tensor=inputs,
                         input_shape=(config["scan"]["size_w"], config["scan"]["size_h"], 3),
                         weights='imagenet')
    res_net.trainable = False

    x = res_net(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(100, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(2, activation='softmax')(x)

    resnet_model = Model(inputs, outputs)

    resnet_model.compile(optimizer=Adam(learning_rate=1e-2),
                         loss=['categorical_crossentropy'],
                         metrics=['accuracy'])

    for layers in resnet_model.layers[-4:]:
        layers.trainable = True

    return resnet_model


def define_model(channels, config):
    classes = config["hyperparameter"]["classes"]
    layers = config["hyperparameter"]["layers"]
    optimizer = config["hyperparameter"]["optimizer"]
    loss = config["hyperparameter"]["loss"]
    metric = config["hyperparameter"]["accuracy"]
    activation = config["hyperparameter"]["activation"]
    activation_dense = config["hyperparameter"]["activation_dense"]
    kernel_init = config["hyperparameter"]["kernel_initializer"]
    filter_init = config["hyperparameter"]["filter_init"]
    filter_size = config["hyperparameter"]["filter_size"]
    pool_size = config["hyperparameter"]["pool_size"]
    padding = config["hyperparameter"]["padding"]
    l2 = config["hyperparameter"]["l2"]
    lr_sgd = config["hyperparameter"]["lr_sgd"]
    lr_adam = config["hyperparameter"]["lr_adam"]
    momentum = config["hyperparameter"]["momentum"]
    w = config["scan"]["size_w"]
    h = config["scan"]["size_h"]

    model = Sequential()

    for i in range(layers):
        filters = filter_init * i
        conv_name = "conv_" + str(i + 1)
        bn_name = "BN_" + str(i + 1)

        model.add(Conv2D(filters,
                         (filter_size, filter_size),
                         activation=activation,
                         kernel_initializer=kernel_init,
                         input_shape=(w, h, channels),
                         padding=padding,
                         name=conv_name))
        model.add(BatchNormalization(name=bn_name))

        if i < (layers - 1):
            model.add(MaxPool2D((pool_size, pool_size)))
        else:
            model.add(Flatten())

    model.add(Dense(classes,
                    activation=activation_dense,
                    kernel_regularizer=regularizers.l2(l2),
                    name='dense'))

    # compile model
    if optimizer == "sgd":
        opt = SGD(learning_rate=lr_sgd, momentum=momentum)
    else:
        opt = Adam(learning_rate=lr_adam)
    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    return model


def define_model_3d(depth, config):
    classes = config["hyperparameter"]["classes"]
    layers = config["hyperparameter"]["layers"]
    optimizer = config["hyperparameter"]["optimizer"]
    loss = config["hyperparameter"]["loss"]
    metric = config["hyperparameter"]["accuracy"]
    activation = config["hyperparameter"]["activation"]
    activation_dense = config["hyperparameter"]["activation_dense"]
    kernel_init = config["hyperparameter"]["kernel_initializer"]
    filter_init = config["hyperparameter"]["filter_init"]
    filter_size = config["hyperparameter"]["filter_size"]
    pool_size = config["hyperparameter"]["pool_size"]
    padding = config["hyperparameter"]["padding"]
    l2 = config["hyperparameter"]["l2"]
    lr_sgd = config["hyperparameter"]["lr_sgd"]
    lr_adam = config["hyperparameter"]["lr_adam"]
    momentum = config["hyperparameter"]["momentum"]
    w = config["scan"]["size_w"]
    h = config["scan"]["size_h"]

    model = Sequential()

    for i in range(layers):
        filters = filter_init * i
        conv_name = "conv_" + str(i + 1)
        bn_name = "BN_" + str(i + 1)

        model.add(Conv3D(filters,
                         (filter_size, filter_size, filter_size),
                         activation=activation,
                         kernel_initializer=kernel_init,
                         input_shape=(w, h, depth, 1),
                         padding=padding,
                         name=conv_name))
        model.add(BatchNormalization(name=bn_name))

        if i < (layers - 1):
            model.add(MaxPool3D((pool_size, pool_size, pool_size)))
        else:
            model.add(Flatten())

    model.add(Dense(classes,
                    activation=activation_dense,
                    kernel_regularizer=regularizers.l2(l2),
                    name='dense'))

    # compile model
    if optimizer == "sgd":
        opt = SGD(learning_rate=lr_sgd, momentum=momentum)
    else:
        opt = Adam(learning_rate=lr_adam)
    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    return model


def define_model_site(config):
    classes = config["hyperparameter"]["classes"]
    activation_dense = config["hyperparameter"]["activation_dense"]
    loss = config["hyperparameter"]["loss"]
    metric = config["hyperparameter"]["accuracy"]
    optimizer = config["hyperparameter"]["optimizer"]
    lr_sgd = config["hyperparameter"]["lr_sgd"]
    lr_adam = config["hyperparameter"]["lr_adam"]
    momentum = config["hyperparameter"]["momentum"]

    model = load_model("D:\\Lab\\CNN-Project\\models\\exp_6\\slice_84_7_channels_run_81.h5")
    model.trainable = False

    x = model.layers[-2].output
    x = Dense(classes, activation=activation_dense)(x)
    model_new = Model(inputs=model.input, outputs=x)

    # compile model
    if optimizer == "sgd":
        opt = SGD(learning_rate=lr_sgd, momentum=momentum)
    else:
        opt = Adam(learning_rate=lr_adam)
    model.compile(optimizer=opt, loss=loss, metrics=[metric])

    return model_new
