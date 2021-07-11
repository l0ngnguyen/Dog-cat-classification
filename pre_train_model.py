
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
import os

path = 'C:\\Users\\vanlo\\vscode_workspace\\Dataset\\car\\car_ims\\000001.jpg'
model = VGG16()
model.get_weights()

#plot_model(model, to_file='vgg.png')


def predict_image(path):
    image = load_img(path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    yhat = model.predict(image)
    label = decode_predictions(yhat)
    label = label[0][0]
    return (label[1], round(label[2]*100, 2))


def predict_set_image(folder_path, image):
    result = {}
    for e in image:
        path = folder_path + e
        result[e] = predict_image(path)
    return result


#listfile = os.listdir('C:\\Users\\vanlo\\vscode_workspace\\vgg16\\data\\')
#print(predict_set_image('C:\\Users\\vanlo\\vscode_workspace\\vgg16\\data\\', listfile))
print(predict_image('D:/img.jpg'))
