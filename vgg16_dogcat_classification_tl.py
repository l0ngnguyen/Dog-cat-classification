from vgg16_dogcat_classification import get_generator, train
import pandas as pd
from tensorflow import keras
import argparse

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

model = create_model()

df =  pd.read_csv(args.dataset_df_path)
df['category'] = df['category'].astype(str)

training_gen, testing_gen = get_generator(df, args.training_path, IMAGE_SIZE, batch_size=args.batch_size)
train(model, training_gen, testing_gen, epochs=args.epochs)

model.save('vgg16_dogcat.h5')
