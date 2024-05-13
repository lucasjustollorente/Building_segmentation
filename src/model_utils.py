from data_utils import filter_images_simple, filter_images_by_white_percentage, calculate_white_percentage
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import segmentation_models as sm
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
#%env SM_FRAMEWORK=tf.keras

# Función para definir el modelo U-Net desde cero
def define_custom_unet(height_shape, width_shape):
    # Definimos la entrada al modelo
    Image_input = Input((height_shape, width_shape, 3))
    Image_in = Lambda(lambda x: x / 255)(Image_input)

    #contracting path
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(Image_in)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    maxp1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(maxp1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    maxp2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(maxp2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    maxp3 = MaxPooling2D((2, 2))(conv3)
    
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(maxp3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    maxp4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(maxp4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    #expansive path
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    
    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    
    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    
    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
 


    model = Model(inputs=[Image_input], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Función para entrenar un modelo U-Net personalizado
def train_custom_unet(X_train, Y_train, X_test, Y_test, height_shape, width_shape, epochs):
    model = define_custom_unet(height_shape, width_shape)

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

    results = model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
                        batch_size=16, epochs=epochs, callbacks=[tensorboard_callback])
    return model, results

# Función para entrenar un modelo U-Net preentrenado usando segmentation_models
def train_pretrained_unet(X_train, Y_train, X_test, Y_test, backbone, epochs):

    model = sm.Unet(backbone, classes=1, activation='sigmoid', encoder_weights='imagenet')

    preprocess_input = sm.get_preprocessing(backbone)
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    model.compile(optimizer='Adam', loss=sm.losses.bce_jaccard_loss, metrics=['accuracy'])

    results = model.fit(x=X_train, y=Y_train, batch_size=16, epochs=epochs, validation_data=(X_test, Y_test))
    return model, results

def predict_images(model, X_val):
    preds = model.predict(X_val)
    return preds

def display_images_in_rows(images_rows, titles_rows=None):
    num_rows = len(images_rows)
    fig, axs = plt.subplots(num_rows, 3, figsize=(15, 5*num_rows))
    
    for i in range(num_rows):
        for j in range(3):
            axs[i, j].imshow(images_rows[i][j])
            axs[i, j].axis('off')
            if titles_rows and i < len(titles_rows) and j < len(titles_rows[i]):
                axs[i, j].set_title(titles_rows[i][j])
    
    plt.show()