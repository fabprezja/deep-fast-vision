# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import tensorflow as tf
from tensorflow.keras.layers import Flatten, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.models import Model

def get_preprocess_input_function(model_name):
    """
    Returns the preprocessing function associated with the given transfer model name.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        function: The preprocess_input function associated with the given transfer model.
    """
    preprocess_input_module = getattr(tf.keras.applications, model_name.lower())
    return preprocess_input_module.preprocess_input

def create_base_model(transfer_arch, image_size, pre_trained):
    """
    Create a base model using the specified transfer architecture and pretrained weights.

    Args:
        transfer_arch (str): Name of the transfer architecture (e.g., 'VGG19', 'ResNet50', etc.).
        image_size (tuple): Tuple containing the height and width of the input image (e.g., (224, 224)).
        pre_trained (str): Name of the dataset used to pre-train the model (e.g., 'imagenet').

    Returns:
        Tuple[Model, function]: A tuple containing the base model and the preprocess input function.
    """
    try:
        base_model_class = getattr(tf.keras.applications, transfer_arch)
        preprocess_input = get_preprocess_input_function(transfer_arch)
    except AttributeError:
        raise ValueError(f"Invalid transfer architecture. "
                         f"Please check the name of the transfer architecture: {transfer_arch}")

    base_model = base_model_class(weights=pre_trained, include_top=False,
                                  input_shape=(image_size[0], image_size[1], 3))
    return base_model, preprocess_input


def create_model ( base_model , before_dense , regularization , number_of_targets , dense_make , initializer ,
                   batch_norm , l1_strength , l2_strength , dropout_rate , activations):
    """
    Creates a model based on the provided arguments.

    Args:
        base_model (tf.keras.Model): The base model for transfer learning.
        before_dense (str): The layer type to add before dense layers ('Flatten' or 'GlobalAveragePooling2D').
        regularization (str): The regularization method to apply ('L1', 'L2', 'Dropout+L1', 'Dropout+L2', or None).
        number_of_targets (int): The number of target classes.
        dense_make (list): A list of integers representing the number of neurons in each dense layer.
        initializer (str): The initializer to use for dense layers.
        batch_norm (bool): Whether to use batch normalization in the model.
        l1_strength (float): The L1 regularization strength.
        l2_strength (float): The L2 regularization strength.
        dropout_rate (float): The dropout rate for dropout regularization.
        activations (str): The activation function to use for dense layers.


    Returns:
        tf.keras.Model: The created model based on the provided arguments.
    """
    if before_dense not in ('Flatten' , 'GlobalAveragePooling2D'):
        raise ValueError ( "Invalid before_dense value. Supported values: 'Flatten', "
                           "'GlobalAveragePooling2D'" )

    if regularization not in ('L1' , 'L2' ,'Dropout', 'Dropout+L1' , None , 'L2+Dropout'):
        raise ValueError ( "Invalid regularization value. Supported values: 'L1', 'L2',"
                           "'Dropout+L1', None, 'Dropout+L2'" )

    before_dense_layer = Flatten if before_dense == 'Flatten' else GlobalAveragePooling2D
    x = before_dense_layer ( ) ( base_model.output )

    for i , neurons in enumerate ( dense_make ):
        if batch_norm:

            x = BatchNormalization ( ) ( x )
        if regularization == 'L1':
            x = Dense ( neurons , activation=activations , kernel_initializer=initializer ,
                        kernel_regularizer=l1 ( l1_strength ) ) ( x )
        elif regularization == 'L2':
            x = Dense ( neurons , activation=activations , kernel_initializer=initializer ,
                        kernel_regularizer=l2 ( l2_strength ) ) ( x )
        elif regularization == 'Dropout':
            x = Dropout ( dropout_rate ) ( x )
            x = Dense ( neurons , activation=activations , kernel_initializer=initializer ) ( x )
        elif regularization == 'Dropout+L1':
            x = Dropout ( dropout_rate ) ( x )
            x = Dense ( neurons , activation=activations , kernel_initializer=initializer ,
                        kernel_regularizer=l1 ( l1_strength ) ) ( x )
        elif regularization == 'Dropout+L2':
            x = Dropout ( dropout_rate ) ( x )
            x = Dense ( neurons , activation=activations , kernel_initializer=initializer ,
                        kernel_regularizer=l2 ( l2_strength ) ) ( x )
        elif regularization is None:
            x = Dense ( neurons , activation=activations ) ( x )

    if number_of_targets == 2:
        activation = 'sigmoid'
        output_units = 1
    else:
        activation = 'softmax'
        output_units = number_of_targets

    x = Dense ( output_units , activation=activation ) ( x )
    model = Model ( inputs=base_model.input , outputs=x )
    return model