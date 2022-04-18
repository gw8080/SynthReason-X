import torch
import random
import transformers
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import cv2
import pyautogui
import random
from time import sleep
from datetime import datetime
transformers.logging.set_verbosity_error()
trainNum = 10
print("AI-Synthetic dawn");
optionB = input ("Do you want to download some sample pictures [yes/no]?:")
if optionB == "yes":
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
    data_dir = pathlib.Path(data_dir)
random.seed(datetime.now())
db_path = "C:\\Users\\georg\\.keras\\datasets\\flower_photos\\" #change and create folder before running AI
db2_path = "C:\\Users\\georg\\.keras\\datasets\\flower_photos\\" #change and create folder before running AI
option = input ("Do you want to: load the vision model or train the vision model [load/train]?:")#select vision model option
#optionB = input ("Do you want to SynthReason to learn [yes/no]?:")#select learning mode activation
#Vision model stuff
data_dir = pathlib.Path(db_path)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)
batch_size = 32
img_height = 180
img_width = 180
train_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.4,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.4,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)
class_names = train_ds.class_names
print(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
num_classes = len(class_names)
modelV = Sequential([
layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
layers.Conv2D(16, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(32, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Conv2D(64, 3, padding='same', activation='relu'),
layers.MaxPooling2D(),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dense(num_classes)
])
modelV.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#Vision model stuff
if option == "train":#train vision model    
    epochs=int(input("epochs:"))
    history = modelV.fit(
train_ds,
validation_data=val_ds,
epochs=epochs
    )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    modelV.save("vision_model")
if option == "load":#load vision model
    modelV = keras.models.load_model("vision_model")
    cap = cv2.VideoCapture(0)
i = 0
while(True):#main loop
    print("-----------------------------------------------------------------------------------")
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image),
                 cv2.COLOR_RGB2BGR)
    cv2.imwrite(db2_path + "\\frame.png", image)
    sunflower_path = db2_path + "\\frame.png"
    img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    #take desktop screenshot
    predictions = modelV.predict(img_array)#make prediction
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    var = round(100 * np.max(score))
    prev = ""
    action = round(random.uniform(0, 100))#do random action or load action from agency
    proc = "I am doing random action " + str(action) + " with " +  class_names[np.argmax(score)]
    prev = class_names[np.argmax(score)] + "=" + str(action)
    print(proc.rstrip())
    pathX = db_path + prev + "_" + str(round(100 * np.max(score))) + "\\"
    image = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(image),
                cv2.COLOR_RGB2BGR)
    isExist = os.path.exists(pathX)
    if not isExist:
        os.makedirs(pathX)
    cv2.imwrite(pathX + str(i) + ".png",image)
    img = tf.keras.utils.load_img(
    pathX + str(i) + ".png", target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = modelV.predict(img_array)#make prediction
    score = tf.nn.softmax(predictions[0])
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )
    if round(100 * np.max(score)) > var:
        file = open('agency.txt', 'a')
        file.write(pathX + '\\' + prev + "_" + str(round(100 * np.max(score))) + "\n")
        file.close()   
    if round(100 * np.max(score)) < var:
        print("No improvement")
    with open('agency.txt') as f:
        for line in list(f):
            if line.find(str(class_names[np.argmax(score)]) + "=") != -1 and line.find( "=" + str(round(100 * np.max(score))) + "_" ) != -1:
                print("Doing purposeful action in: " + line)
                break
    i+=1