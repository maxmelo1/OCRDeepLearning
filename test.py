import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7, ResNet50V2, ResNet101V2, ResNet152V2, VGG16, VGG19, InceptionResNetV2
from tensorflow.keras.preprocessing.image import load_img

from PIL import Image
import numpy as np
import argparse
import os

DATASET_PATH = 'English/Img/GoodImg/Bmp'
IMG_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 62

def predict(img_path):
    model = model = keras.models.load_model('saved/model/model_new.h5')
    #model.summary()

    # image = Image.open(img_path).convert("RGB")
    # #print(image)
    # image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
    # image = np.array(image)
    # image = image / 255.
    # image = np.expand_dims(image, axis=0)

    image  = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    image  = tf.keras.preprocessing.image.img_to_array(image)
    #image  /= 255.
    image = np.expand_dims(image, axis=0)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode="categorical",
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = val_ds.class_names

    sample = next(iter(val_ds))

    
    result = model.predict(image)

    print(f'predicted as:{class_names[np.argmax(result, axis=-1)[0]]}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing Classification CNN')
    parser.add_argument('-img', action="store", required=True, help='image path', dest='img_path')

    arguments = parser.parse_args()
    
    img_path = DATASET_PATH+arguments.img_path
    predict(img_path)