import argparse
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
import tensorflow as tf
from keras import layers
from keras.layers import Conv2D
from keras.layers import MaxPool2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.test_ops import binary
from tensorflow.python.ops.random_ops import categorical
from gc import callbacks
from tensorflow._api.v2 import train
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parse = argparse.ArgumentParser(description='Deep learning training',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parse.add_argument('dataset_df_path', help='input the dataset dataframe')
parse.add_argument('training_path', help='the training path store all images')
parse.add_argument('image_size', help='input the height or width of image', default=190, type=int)
parse.add_argument('--batch_size', help='the batch size when training', type=int, default=10)
parse.add_argument('--epochs', help='number of epochs', type=int, default=10)
args = parse.parse_args()

IMAGE_WIDTH = IMAGE_HEIGHT = args.image_size
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


# Build Model
def build_model_v1():
    # sequential model like an list in python with element is layer
    # add layers
    model = Sequential()
    model.add(Conv2D(input_shape=INPUT_SHAPE, filters=64,
                     kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dense(units=2, activation='softmax'))

    # set optimizer, loss,...
    model.compile(optimizer=Adam(lr=0.001),
                  loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    return model


def build_model_v2():
    model = Sequential()
    model.add(Conv2D(input_shape=INPUT_SHAPE, filters=64,
                     kernel_size=(3, 3), padding="same", activation="relu", ))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(
        3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    op = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=op,
                  metrics=['accuracy'])
    
    return model


# Preprocessing
# Read file

def get_generator(df, train_path, image_size, test_size=0.2, batch_size=10):
    df_training, df_testing = train_test_split(
        df, test_size=test_size, random_state=42)
    df_training = df_training.reset_index(drop=True)
    df_testing = df_testing.reset_index(drop=True)

    train_generator = ImageDataGenerator()
    training_set = train_generator.flow_from_dataframe(
        dataframe=df_training,
        directory=train_path,
        x_col='file name',
        y_col='category',
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )

    testing_set = train_generator.flow_from_dataframe(
        dataframe=df_testing,
        directory=train_path,
        x_col='file name',
        y_col='category',
        target_size=image_size,
        class_mode='binary',
        batch_size=batch_size,
        shuffle=True
    )
    return training_set, testing_set


def plot_train_history(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def train(model, training_gen, testing_gen, epochs=10):
    # To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased
    # stop the learning after 10 epochs and val_loss value not decreased
    earlystop = EarlyStopping(patience=10)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',  # reduce the learning rate when then accuracy not increase for 2 steps
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]

    hist = model.fit_generator(generator=training_gen,
                               validation_data=testing_gen,
                               epochs=epochs,
                               callbacks=callbacks)
    plot_train_history(hist)



if __name__ == '__main__':
    CLASS_NAMES = ('cat', 'dog')
    train_path = 'input/train/'
    test_path = 'input/test1/'

    #df = readfile(train_path)
    #df.to_csv('dog_cat_train.csv', index=False)

    model = build_model_v2()
    model.summary()

    df =  pd.read_csv(args.dataset_df_path)
    df['category'] = df['category'].astype(str)

    training_gen, testing_gen = get_generator(df, args.training_path, IMAGE_SIZE, batch_size=args.batch_size)

    train(model, training_gen, testing_gen, epochs=args.epochs)

