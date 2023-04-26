# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def variable_gaussian_blur ( img , max_radius=3 ):
    """
    Apply Gaussian blur to an image with a random radius in the range [0, max_radius].

    Args:
        img (np.array): Image to be blurred.
        max_radius (int, optional): Maximum blur radius. Defaults to 3.

    Returns:
        np.array: Blurred image.
    """
    blur_radius = np.random.uniform ( 0 , max_radius )
    pil_image = Image.fromarray ( img.astype ( np.uint8 ) )
    blurred_image = pil_image.filter ( ImageFilter.GaussianBlur ( radius=blur_radius ) )
    return np.array ( blurred_image )

def create_image_data_generator ( augmentation , custom_augmentation , preprocess_input ):
    """
    Create an ImageDataGenerator for the specified augmentation type and custom augmentation function.

    Args:
        augmentation (str): Type of augmentation ('no_aug', 'basic', 'advanced', 'advanced_with_blur', or 'custom').
        custom_augmentation (function, optional): Custom augmentation function.
        preprocess_input (function): Preprocessing function for the input images.

    Returns:
        ImageDataGenerator: An ImageDataGenerator instance for the specified augmentation and preprocessing.
    """
    def preprocess_input_with_blur ( img ):
        img = preprocess_input ( img )
        img = variable_gaussian_blur ( img )
        return img

    def custom_aug_with_preprocess_input ( img ):
        img = preprocess_input ( img )
        img = custom_augmentation ( img )
        return img

    augmentation_options = {
        'no_aug': ImageDataGenerator (
            preprocessing_function=preprocess_input
        ) ,
        'basic': ImageDataGenerator (
            preprocessing_function=preprocess_input ,
            shear_range=0.2 ,
            zoom_range=0.2 ,
            horizontal_flip=True
        ) ,
        'advanced': ImageDataGenerator (
            preprocessing_function=preprocess_input ,
            rotation_range=40 ,
            width_shift_range=0.2 ,
            height_shift_range=0.2 ,
            shear_range=0.2 ,
            zoom_range=0.2 ,
            horizontal_flip=True ,
            fill_mode='nearest'
        ) ,
        'advanced_with_blur': ImageDataGenerator (
            preprocessing_function=preprocess_input_with_blur ,
            rotation_range=40 ,
            width_shift_range=0.2 ,
            height_shift_range=0.2 ,
            shear_range=0.2 ,
            zoom_range=0.2 ,
            horizontal_flip=True ,
            fill_mode='nearest'
        ) ,
        'custom': ImageDataGenerator (
            preprocessing_function=custom_aug_with_preprocess_input
        )
    }

    if augmentation not in augmentation_options and not (augmentation == 'custom' and custom_augmentation is not None):
        raise ValueError (
            "Invalid augmentation. Supported options: 'no_aug', 'basic', 'advanced', 'advanced_with_blur', 'custom'" )
    return augmentation_options [ augmentation ]