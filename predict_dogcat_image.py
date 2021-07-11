from tensorflow import keras
from PIL import Image
import numpy as np
from vgg16_dogcat_classification import CLASS_NAMES
import os
import matplotlib.pyplot as plt
import math

load_model = keras.models.load_model('E:/vgg16_dogcat.h5')

def predict_single_image(img):
    img = img.resize((190, 190))
    img = np.asarray(img)
    img = np.expand_dims(img, 0)
    return int(load_model.predict(img)[0][0])
def predict_image(folder_path, size):
    filenames = os.listdir(folder_path)[:size]
    for i in range(size):
        img = Image.open(folder_path + filenames[i])
        c = predict_single_image(img)
        plt.subplot(math.ceil(size/5), 5, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(CLASS_NAMES[c])
    plt.show()
    
predict_image('C:/Users/vanlo/vscode_workspace/Dataset/dog_cat_image/test1/', 22)
    