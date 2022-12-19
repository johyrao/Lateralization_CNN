import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD, Adam, schedules
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import regularizers, Input
from sklearn import svm, linear_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc, confusion_matrix


