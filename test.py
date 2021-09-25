import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7, ResNet50V2, ResNet101V2, ResNet152V2, VGG16, VGG19, InceptionResNetV2
from tensorflow.keras.preprocessing.image import load_img
from keras.utils.data_utils import get_file

from PIL import Image
import numpy as np
import argparse
import os

DATASET_PATH = 'English/Img/GoodImg/Bmp'
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 62

WEIGHTS_PATH = 'https://github.com/maxmelo1/OCRDeepLearning/releases/download/0.1/model_new.h5'

class_names = ["Sample00"+str(i) if i<10 else "Sample0"+str(i) for i in range(1,63)]

def predict(img_path):
    weights_path = get_file(
            'resnet50v2_ocr_natural_images.h5',
            WEIGHTS_PATH,
            cache_subdir='models')
    model = model = keras.models.load_model(weights_path)
    #model.summary()

    image  = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    image  = tf.keras.preprocessing.image.img_to_array(image)
    #image  /= 255.
    image = np.expand_dims(image, axis=0)
    
    result = model.predict(image)
    pred = np.argmax(result, axis=-1)[0]

    print(f'predicted as:{class_names[pred]}')
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    fig.subplots_adjust(top=0.85)

    plt.imshow(image.squeeze().astype("uint8"))
    plt.title(f'Predicted as [{class_names[pred]}]')
    
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing Classification CNN')
    parser.add_argument('-img', action="store", required=True, help='image path', dest='img_path')

    arguments = parser.parse_args()
    
    img_path = DATASET_PATH+arguments.img_path
    predict(img_path)