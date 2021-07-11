from vgg16_dogcat_classification import get_generator, train, plot_train_history
import pandas as pd
from tensorflow import keras
from tensorflow.python.framework.test_ops import binary
from tensorflow.python.ops.random_ops import categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

IMAGE_WIDTH = IMAGE_HEIGHT = 190
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
INPUT_SHAPE = (IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)

def create_model():
    trained = keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=INPUT_SHAPE
    )
    trained.summary()
    trained.trainable = False

    model = keras.models.Sequential()
    model.add(trained)
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    op = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        loss='binary_crossentropy',
        optimizer=op,
        metrics=['accuracy']
    )
    return model

CLASS_NAMES = ('cat', 'dog')
train_path = 'input/train/'
test_path = 'input/test1/'

model = create_model()

df =  pd.read_csv('dog_cat_train.csv')
df['category'] = df['category'].astype(str)

training_gen, testing_gen = get_generator(df, train_path, IMAGE_SIZE)
train(model, training_gen, testing_gen, epochs=10)

model.save('vgg16_dogcat.h5')
