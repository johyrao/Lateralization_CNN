import numpy as np
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D, \
    Dropout, Conv3D, MaxPooling3D
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


def make_resnet(opt):
    # Make pretrain resnet model
    inputs = Input(shape=(opt["scan"]["size_w"], opt["scan"]["size_h"], 3))

    res_net = ResNet50V2(include_top=False, input_tensor=inputs,
                         input_shape=(opt["scan"]["size_w"], opt["scan"]["size_h"], 3),
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


def define_model_sgd(channels, config):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     input_shape=(config["scan"]["size_w"], config["scan"]["size_h"], channels),
                     padding="same", name='conv2d_1'))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv2d_2'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv2d_3'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(.001)))

    # compile model
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def define_model_adam(channels, config):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     input_shape=(config["scan"]["size_w"], config["scan"]["size_h"], channels),
                     padding="same", name='conv_1'))
    model.add(BatchNormalization(name='BN_1'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv_2'))
    model.add(BatchNormalization(name='BN_2'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv_3'))
    model.add(BatchNormalization(name='BN_3'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(.0001), name='dense'))

    # compile model
    # lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=1000,decay_rate=0.9)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def define_model_site(channels, config):
    model = Sequential()
    model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform',
                     input_shape=(config["scan"]["size_w"], config["scan"]["size_h"], channels),
                     padding="same", name='conv_1'))
    model.add(BatchNormalization(name='BN_1'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv_2'))
    model.add(BatchNormalization(name='BN_2'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv_3'))
    model.add(BatchNormalization(name='BN_3'))
    model.add(Flatten())
    model.add(Dense(6, activation='softmax', kernel_regularizer=regularizers.l2(.0001), name='dense'))

    # compile model
    # lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=1000,decay_rate=0.9)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def define_model_3d(depth, config):
    model = Sequential()
    model.add(Conv3D(8, (3, 3, 3), activation='relu', kernel_initializer='he_uniform',
                     input_shape=(config["scan"]["size_w"], config["scan"]["size_h"], depth),
                     padding="same", name='conv_1'))
    model.add(BatchNormalization(name='BN_1'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv_2'))
    model.add(BatchNormalization(name='BN_2'))
    model.add(MaxPooling3D((2, 2, 2)))
    model.add(Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer='he_uniform', padding="same", name='conv_3'))
    model.add(BatchNormalization(name='BN_3'))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(.0001), name='dense'))

    # compile model
    # lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=1000,decay_rate=0.9)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
