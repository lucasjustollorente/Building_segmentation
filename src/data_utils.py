import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import datetime

# Librerias para constuir la arquitectura U-Net
from tensorflow.keras.layers import Input, Lambda, Conv2D, Dropout, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model



def calculate_white_percentage(image):
    """Calcula el porcentaje de píxeles blancos en una imagen."""
    total_pixels = image.shape[0] * image.shape[1]
    white_pixels = np.sum(image == 255)  # Contar píxeles blancos
    return white_pixels / total_pixels


def filter_images_by_white_percentage(data_path_images, data_path_masks, height_shape, width_shape, white_threshold):
    """Filtrar imágenes y seleccionar máscaras basadas en el porcentaje de píxeles blancos."""
    data_list_images = os.listdir(data_path_images)
    filtered_images = []
    filtered_masks = []

    for file_name in tqdm(data_list_images):
        # Leer la imagen
        img = imread(os.path.join(data_path_images, file_name))[:,:,:3]  
        # Calcular el porcentaje de píxeles blancos
        white_percentage = calculate_white_percentage(img)
        
        if white_percentage <= white_threshold:
            # Redimensionar la imagen
            img = resize(img, (height_shape, width_shape), mode='constant', preserve_range=True)
            filtered_images.append(img)

            # Obtener el nombre de la máscara correspondiente
            mask_file = os.path.splitext(file_name)[0] + '.png'  # Nombre de la máscara
            mask_path = os.path.join(data_path_masks, mask_file)
            if os.path.exists(mask_path):
                # Leer y redimensionar la máscara
                maskt = imread(mask_path)[:,:,1]
                maskt = resize(maskt, (height_shape, width_shape), mode='constant', preserve_range=True)
                filtered_masks.append(maskt)

    # Convertir las listas en arrays NumPy
    filtered_images = np.asarray(filtered_images, dtype=np.uint8)
    filtered_masks = np.asarray(filtered_masks, dtype=bool)

    return filtered_images, filtered_masks


def filter_images_simple(data_path_images, data_path_masks, height_shape, width_shape):
    """Filtrar imágenes y seleccionar máscaras sin ningún criterio."""
    data_list_images = os.listdir(data_path_images)
    filtered_images = []
    filtered_masks = []

    for file_name in tqdm(data_list_images):
        # Leer la imagen
        img = imread(os.path.join(data_path_images, file_name))[:,:,:3]  
        # Redimensionar la imagen
        img = resize(img, (height_shape, width_shape), mode='constant', preserve_range=True)
        filtered_images.append(img)

        # Obtener el nombre de la máscara correspondiente
        mask_file = os.path.splitext(file_name)[0] + '.png'  # Nombre de la máscara
        mask_path = os.path.join(data_path_masks, mask_file)
        if os.path.exists(mask_path):
            # Leer y redimensionar la máscara
            maskt = imread(mask_path)[:,:,1]
            maskt = resize(maskt, (height_shape, width_shape), mode='constant', preserve_range=True)
            filtered_masks.append(maskt)

    # Convertir las listas en arrays NumPy
    filtered_images = np.asarray(filtered_images, dtype=np.uint8)
    filtered_masks = np.asarray(filtered_masks, dtype=bool)

    return filtered_images, filtered_masks

