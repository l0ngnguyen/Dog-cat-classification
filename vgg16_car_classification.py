from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.python.framework.test_ops import binary
from tensorflow.python.ops.random_ops import categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMAGE_WIDTH = 190
IMAGE_HEIGHT = 190
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)


def create_model():
    trained = keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=INPUT_SHAPE
    )
    trained.trainable = False
    model = keras.models.Sequential()
    model.add(trained)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=512, activation="relu"))
    model.add(keras.layers.Dropout(0.7))
    model.add(keras.layers.Dense(units=196, activation="softmax"))

    op = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='categorical_crossentropy',
                  optimizer=op,
                  metrics=['accuracy'])
    return model


def preprocessing_data():
    path = 'C:/Users/vanlo/vscode_workspace/Dataset/car/'
    df = pd.read_csv(path + 'cars.csv')
    df['class'] = df['class'].astype(str)
    #df = df.loc[:6000, :]
    df_training, df_testing = train_test_split(
        df, test_size=0.20, random_state=42)
    df_training = df_training.reset_index(drop=True)
    df_testing = df_testing.reset_index(drop=True)

    train_generator = ImageDataGenerator()
    training_set = train_generator.flow_from_dataframe(
        dataframe=df_training,
        directory=path,
        x_col='relative_im_path',
        y_col='class',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=10
    )

    testing_set = train_generator.flow_from_dataframe(
        dataframe=df_testing,
        directory=path,
        x_col='relative_im_path',
        y_col='class',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=10
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


def main():
    model = create_model()
    model.summary()
    training_set, testing_set = preprocessing_data()
    history = model.fit_generator(generator=training_set,
                                  validation_data=testing_set,
                                  epochs=10)
    plot_train_history(history)


main()
