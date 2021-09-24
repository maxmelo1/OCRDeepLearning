import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB7, ResNet50V2, ResNet101V2, ResNet152V2, VGG16, VGG19, InceptionResNetV2
import numpy as np
import wandb
from wandb.keras import WandbCallback

path_dataset = 'English/Img/GoodImg/Bmp'
IMG_SIZE = 64
batch_size = 32
NUM_CLASSES = 62

epochs = 100
LR = 1e-5

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_dataset,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_dataset,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=batch_size
)

class_names = train_ds.class_names

# plt.figure(figsize=(10, 10))
# for images, labels in train_ds.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(images[i].numpy().astype("uint8"))
#         plt.title(class_names[int(labels[i])])
#         plt.axis("off")

    #plt.show()



data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        #layers.experimental.preprocessing.RandomFlip("vertical"),
        layers.experimental.preprocessing.RandomRotation(0.1),
        layers.experimental.preprocessing.RandomContrast(0.2),
    ]
)
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")
#     plt.show()


def input_preprocess(image, label):
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

size = (IMG_SIZE, IMG_SIZE)
train_ds = train_ds.map(lambda image, label: (tf.image.resize(image, size), label))
val_ds = val_ds.map(lambda image, label: (tf.image.resize(image, size), label))

train_ds = train_ds.map(input_preprocess)
val_ds = val_ds.map(input_preprocess)

def build_model(num_classes):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = layers.Rescaling(1./255)(x)

    model = ResNet50V2(include_top=False, input_tensor=inputs, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False

    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.25
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name="ResNet50")
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def unfreeze_model(model):
    for layer in model.layers:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


wandb.login()

wandb.init(
      # Set entity to specify your username or team name
      # ex: entity="carey",
      # Set the project where this run will be logged
      project="ocr-natural-images", 
      # Track hyperparameters and run metadata
      config={
      "learning_rate": LR,
      "architecture": "ResNet50V2",
      "dataset": "Chars74K",
      "hyper": "parameter",})


model = build_model(num_classes=NUM_CLASSES)
model = unfreeze_model(model)
#model = keras.models.load_model('saved/model/model.h5')

model.summary()

hist = model.fit(train_ds, epochs=epochs, validation_data=val_ds, verbose=1, callbacks=[WandbCallback()])
model.save('saved/model/model_new.h5')