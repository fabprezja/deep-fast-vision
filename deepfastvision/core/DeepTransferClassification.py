# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import glob
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from deepfastvision.model_components.callbacks.custom_save_weights import CustomSaveWeights
from deepfastvision.model_components.model_creation import create_base_model, create_model
from deepfastvision.model_components.preprocessing.preprocessing import create_image_data_generator
from deepfastvision.data_management.data_helpers import update_nested_dict
from deepfastvision.plotting import plot_helpers
from deepfastvision.model_components.prediction_helpers import model_predict
from deepfastvision.model_components.export_helpers import export


class DeepTransferClassification:
    """
    This class represents a deep transfer learning classification experiment with customizable settings.
    It is designed to facilitate and automate the creation, training, and evaluation of various models using optional
    data augmentation, regularization, and metric tracking. The class provides
    default settings which can be updated using keyword arguments for a flexible and customizable
    experiment configuration.
    """
    def __init__(self,**kwargs):
        """
        Initialize the DeepTransferClassification with default settings and given keyword arguments.
        The settings can be updated by providing a dictionary of keyword arguments.
        """
        self.kwargs = self.get_kwargs()
        """
        Return the default settings dictionary for the experiment, including paths, model settings,
        training settings, evaluation settings, saving settings, and miscellaneous settings.
        """
        self.kwargs = update_nested_dict(self.kwargs,kwargs)

    def get_kwargs(self):
        """
         Get the default configuration settings for the model training and evaluation pipeline.

         Returns:
             dict: A dictionary containing the default configuration settings for the pipeline, with the following keys:

             - 'paths':

                 - 'train_val_data': str, path to training and validation data folder, default: 'path_to_train_val_data'

                     .. note:: This is the expected folder configuration:

                         .. code-block:: none

                             train/
                                 class_A/
                                     image_1A.png
                                     image_2A.png
                                 class_B/
                                     image_1B.png
                                     image_2B.png
                             val/
                                 class_A/
                                     image_3A.png
                                     image_4A.png
                                 class_B/
                                     image_3B.png
                                     image_4B.png

                         The library uses the key words, 'train', 'val' and 'test' for the three sets
                         (the same format is used for the testing data), while the external test name may vary

                         If your data is not distributed in this format, you can simply use
                         the DataSplitter from data_management.

                 - 'test_data_folder': str or None, path to test data folder, default: None
                 - 'external_test_data_folder': str or None, path to external test data folder, default: None

             - 'model':

                 - 'image_size': tuple, target image re-size (height, width), default: (224, 224)

                     .. note::

                         These values are used to resize generators images. It is recommended to use
                         the original input size of the chosen transfer model.

                 - 'transfer_arch': str, transfer architecture name, default: 'VGG19'

                     .. note::

                         Various  transfer architectures are available, you can call them by name (e.g. 'ResNet101V2').
                         For a list of available architectures, refer to https://keras.io/api/applications.
                         Don't forget to update the freeze settings from the previous
                         architecture before loading a new one.

                 - 'pre_trained': str, which pre-trained weights to load, default: 'imagenet'
                 - 'before_dense': str, layer type before dense but after
                 transfer architecture layers, default: 'Flatten'

                     .. note::

                         Currently two options exist, Flatten or GlobalAveragePooling2D.

                 - 'dense_layers': list, number of dense layers after transfer
                 architecture and 'before_dense' layer argument, default: [256]

                     .. note::

                         The number of dense layers is equal to the length of the list.

                 - 'dense_activations': str, activation function for dense layers, default: 'elu'

                     .. note::

                         For a list of available activations, refer to https://keras.io/api/layers/activations/

                 - 'initializer': str, dense layer initializer, default: 'he_normal'

                     .. note::

                         For a list of available initializers, refer to https://keras.io/api/layers/initializers/

                 - 'batch_norm': bool, whether to use batch normalization between dense layers, default: False
                 - 'regularization': str, regularization type, default: 'Dropout'

                     .. note::

                         The regularization method to apply ('L1', 'L2', 'Dropout+L1', 'Dropout+L2', or None).

                - 'l1_strength': float, L1 regularization strength, default: 0.001
                - 'l2_strength': float, L2 regularization strength, default: 0.001
                - 'dropout_rate': float, dropout rate, default: 0.21
                - 'freeze_weights': int or None, number of layers to freeze, default: None

                  .. note::

                      If set to None, all layers will be trainable. If set to an integer,
                      the last n layers will be frozen.

                - 'unfreeze_block': list or None, list of blocks or layers to unfreeze, default: ['block5']

                  .. note::

                      If set to None, all layers will be trainable. If set to a list, the specified blocks
                      or layer names (as strings) will be unfrozen, e.g., ['block1', 'block2', 'block3'].

                - 'freeze_up_to': str or None, freeze layers up to a specific block, default: 'flatten'

                  .. note::

                      If set to None, all layers will be trainable. If set to a string, all layers up to the specified
                      block or layer will be frozen. When using both 'freeze_up_to' and 'unfreeze_block',
                      it is possible to selectively unfreeze layers or blocks within the 'freeze_up_to' range.

                - 'show_freeze_status': bool, whether to show layer freeze status, default: True
                - 'number_of_targets': int, number of target classes

                  .. warning::

                      This argument is automatically updated based on the number of classes in the training data.
                      Do not modify it unless you are certain about the consequences.

                - 'target_type': str or None, target type

                  .. warning::

                      This argument is automatically updated based on the number of classes in the training data.
                      Do not change, unless you know what you are doing.

            - 'training':

                - 'epochs': int, number of epochs, default: 12
                - 'batch_size': int, batch size, default: 32
                - 'learning_rate': float, learning rate, default: 2e-5
                - 'optimizer_name': str, optimizer name, default: 'Adam'

                  .. note::

                      Provide name of the optimizer you want to use.
                      For a list of available optimizers, refer to https://keras.io/api/optimizers/

                - 'add_optimizer_params': dict, additional optimizer parameters, default: {}

                    .. note::

                      Add additional parameters to the optimizer other than learning rate.
                      For a list of available parameters, refer to https://keras.io/api/optimizers/

                      For example given the Adam optimizer, you may add:
                      {'beta_1': 0.8, 'beta_2': 0.8, 'epsilon': 1e-05, 'clipnorm': 0.8}
                      Or any parameter shown in the documentation: https://keras.io/api/optimizers/adam/

                - 'class_weights': bool, whether to use class weights, default: True
                - 'metrics': list, list of compatible evaluation metrics, default: ['accuracy']

                    .. note:: For a list of available metrics, refer to https://keras.io/api/metrics/

                - 'augmentation': str, data augmentation type, default: 'basic'

                  .. note:: The available augmentation options are:

                    .. code-block:: python

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

                  The library always retrieves the appropriate pre-processing function for the selected
                  transfer learning architecture, along with the specified augmentation. This includes the
                  'preprocess_input_with_blur' option, which introduces varying levels of blur as an additional
                  augmentation technique. The 'custom' augmentation option enables you to supply your own augmentation
                  without fetching the corresponding pre-processing function for the transfer architecture.

                - 'custom_augmentation': callable or None, custom keras augmentation, default: None

                    .. note:: The user provided augmentation. More about augmentation:https://tinyurl.com/augTFKeras

                    .. warning:: This is only applicable when the 'augmentation' argument is set to 'custom'.

                - 'callback': list or None, list of callbacks, default: None

                    .. note:: User provided callbacks and/or available callbacks from https://keras.io/api/callbacks/

                - 'early_stop': float or None, whether to use early stopping epochs set as
                a fraction of total epochs, default: False

                    .. note:: For example, if there are 100 epochs and early_stop is set to 0.1, early stopping will
                    be triggered if the model does not improve for 10 consecutive epochs.

                - 'warm_pretrain_dense': bool, whether to warm pretrain dense layers, default: True

                    .. note:: Pre-train dense layers with a frozen transfer model, then unfreeze and train as specified
                    (to mitigate destructive effects on unfrozen transfer architecture). It is recommended to use this
                    approach only if there are blocks or layers specified unfrozen in the transfer architecture.

                - 'warm_pretrain_epochs': int, number of warm pretraining epochs, default: 5

            - 'evaluation':

                - 'evaluate_mode': bool, whether to use evaluation mode, default: False

                    .. note:: If True, training will not begin.

                - 'auto_mode': bool, whether to use automatic evaluation, default: True

                    .. note:: If True, final test on the best epoch will be performed automatically.

                - 'preloaded_weights_path': str or None, path to user preloaded weights file, default: None

                    .. note:: If provided, the model will be loaded with the provided weights.

            - 'saving':

                - 'save_weights': bool, whether to save model weights, default: True
                - 'save_weights_folder': str, path to save weights folder, default: 'path_to_save_weights'
                - 'save_best_weights': str, which metric to use to save the best
                weights (if 'all' no metric is used), default: 'val_loss'
                - 'save_weights_silent': bool, whether to silently save weights, default: False

            - 'misc':

                - 'show_summary': bool, whether to display model summary, default: True
                - 'plot_curves': bool, whether to plot validation curves, default: True
                - 'show_min_max_plot': bool, whether to show min-max values within validation curves, default: True

                - 'plot_conf': bool, whether to plot confusion matrix, default: True

                    .. note:: The confusion matrix is normalized (rows), and automaticaly uses label names if available.
                    The matrix also adjusts depending on the number of classes.

        """

        return {
            'paths': {
                'train_val_data': 'path_to_train_val_data',
                'test_data_folder': None,
                'external_test_data_folder': None,
            },
            'model': {
                'image_size': (224, 224),
                'transfer_arch': 'VGG19',
                'pre_trained': 'imagenet',
                'before_dense': 'Flatten',
                'dense_layers': [256],
                'dense_activations': 'elu',
                'initializer': 'he_normal',
                'batch_norm': False,
                'regularization': 'Dropout',
                'l1_strength': 0.001,
                'l2_strength': 0.001,
                'dropout_rate': 0.21,
                'freeze_weights': None,
                'unfreeze_block': ['block5'],
                'freeze_up_to': 'flatten'
                ,'show_freeze_status' :True,
                'number_of_targets' :0,
                'target_type': None,
            },
            'training': {
                'epochs': 12,
                'batch_size': 32,
                'learning_rate': 2e-5,
                'optimizer_name': 'Adam',
                'add_optimizer_params': {},
                'class_weights': True,
                'metrics': ['accuracy'],
                'augmentation': 'basic',
                'custom_augmentation': None,
                'callback': None,
                'early_stop': False,
                'warm_pretrain_dense': True,
                'warm_pretrain_epochs': 5,
            },
            'evaluation': {
                'evaluate_mode': False,
                'auto_mode': True,
                'preloaded_weights_path': None,
            },
            'saving': {
                'save_weights': True,
                'save_weights_folder': 'path_to_save_weights',
                'save_best_weights': 'val_loss',
                'save_weights_silent': False,
            },
            'misc': {
                'show_summary': True,
                'plot_curves': True,
                'show_min_max_plot': True,
                'plot_conf':True,
            },
        }

    def export_configuration(self):
        """
        Return the current settings dictionary for the experiment.
        """
        return self.kwargs

    @staticmethod
    def _get_optimizer(optimizer_name, learning_rate, additional_params=None):
        """
        Get an instance of the optimizer specified by the given name and learning rate.
        Additional parameters can be provided in the additional_params dictionary.

        Args:
            optimizer_name (str): The name of the optimizer.
            learning_rate (float): The learning rate for the optimizer.
            additional_params (dict, optional): A dictionary of additional parameters for the optimizer.

        Returns:
            Optimizer: An instance of the specified optimizer with the given learning rate and additional parameters.
        """
        optimizer_class = getattr(tf.keras.optimizers, optimizer_name)
        if additional_params is None:
            additional_params = {}
        return optimizer_class(learning_rate=learning_rate, **additional_params)

    def _get_metrics(self,metrics_names):
        """
        Create a list of Keras metrics objects based on the given metrics names.

        Args:
            metrics_names (list[str]): A list of metric names to create Keras metric objects for.

        Returns:
            metrics (list[tf.keras.metrics.Metric]): A list of Keras metric objects.
        """
        def _create_metric(name):
            if name.lower() == 'accuracy':
                return (tf.keras.metrics.BinaryAccuracy(name=name) if
                        self.kwargs['model']['number_of_targets'] == 2
                        else tf.keras.metrics.CategoricalAccuracy(name=name))
            return getattr(tf.keras.metrics,name)()

        return [_create_metric(name) for name in metrics_names]

    def _set_metrics(self):
        """
        Set the metrics for the experiment based on the metrics specified in the settings dictionary.
        """
        self.metrics = self._get_metrics(self.kwargs['training']['metrics'])

    def _get_number_of_targets(self):
        """
        Determine the number of unique target classes based on the training data directory.
        Returns the number of unique target classes.
        """
        train_dir = os.path.join(self.kwargs['paths']['train_val_data'],'train')
        unique_classes = set()
        for root,_,files in os.walk(train_dir):
            for file in files:
                unique_classes.add(os.path.basename(root))
        num_targets = len(unique_classes)
        print(f"Number of Targets: ",num_targets)
        return num_targets

    def _set_number_of_targets(self):
        """
        Set the number of target classes in the settings dictionary based on the unique target classes in the training data.
        """
        self.kwargs['model']['number_of_targets'] = self._get_number_of_targets()

    def _get_target_type(self,num_targets):
        """
        Determine the target type based on the number of unique target classes.

        Args:
            num_targets (int): Number of unique target classes.

        Returns:
            str: The target type ('binary' or 'categorical').
        """

        target_type = 'binary' if num_targets == 2 else 'categorical'
        print("Target Type: ",target_type)
        return target_type

    def _set_target_type(self):
        """
        Set the target type in the settings dictionary based on the number of unique target classes.
        """
        num_targets = self.kwargs['model']['number_of_targets']
        target_type = self._get_target_type(num_targets)
        self.kwargs['model']['target_type'] = target_type

    def _get_loss(self):
        """
        Determine the loss type based on the number of unique target classes.

        Returns:
            str: The loss type ('binary_crossentropy' or 'categorical_crossentropy').
        """
        lostype='binary_crossentropy' if self.kwargs['model']['number_of_targets'] == 2 else 'categorical_crossentropy'
        print("Loss: ",lostype)
        return lostype

    def _get_best_weights_file(self):
        """
        Get the file path of the best weights file based on the modification time.

        Returns:
            str: The file path of the best weights file.
        """

        weights_files = glob.glob(os.path.join(self.kwargs['saving']['save_weights_folder'],
                                               'best_weights_epoch_*.h5'))
        best_weights_file = max(weights_files,key=os.path.getmtime) if weights_files else None
        return best_weights_file

    def _auto_evaluate(self):
        """
        Load the best weights before evaluation if the auto_mode setting is enabled.
        """
        if self.kwargs['evaluation']['auto_mode']:
            self._load_best_weights()
            print("Loading the best weights before test...")

    def _load_best_weights(self):
        """
        Load the best weights file into the model.
        """
        best_weights_file = self._get_best_weights_file()
        if best_weights_file:
            print("Loading best weights from:",best_weights_file)
            self.model.load_weights(best_weights_file)
        else:
            raise Exception("Best weights file not found. Please train the model first.")

    def _initialize_base_model_and_fetch_prepro(self):
        """
        Initialize the base transfer model and fetch the preprocessing function.
        """
        self.base_model, self.preprocess_input = create_base_model(
            self.kwargs['model']['transfer_arch'],
            self.kwargs['model']['image_size'],
            self.kwargs['model']['pre_trained'],
        )
    def _create_datagens(self,preprocess_input_func):
        """
        Create image data generators for the experiment based on the given preprocessing function.

        Args:
            preprocess_input_func (function): Preprocessing function for input images.
        """
        self.train_datagen = create_image_data_generator(self.kwargs['training']['augmentation'],
                                                         self.kwargs['training']['custom_augmentation'],
                                                         preprocess_input_func)
        self.val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_func)
        self.test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input_func)

    def _create_generator(self,datagen,data_folder,image_size,batch_size,target_type,color_mode='rgb',shuffle=True):
        """
        Create a generator using the given datagen, data_folder, image_size, batch_size, and target_type.

        Args:
            datagen (ImageDataGenerator): Image data generator instance.
            data_folder (str): Path to the data directory.
            image_size (tuple): Tuple of (height, width) for resizing images.
            batch_size (int): Number of image samples per batch.
            target_type (str): Type of target labels ('binary' or 'categorical').
            color_mode (str, optional): Color mode of the images ('rgb' or 'grayscale'). Defaults to 'rgb'.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            DirectoryIterator: A generator yielding tuples of (x, y) where x is a batch of image data and y is a batch of labels.
        """
        return datagen.flow_from_directory(
            data_folder,
            target_size=image_size,
            batch_size=batch_size,
            class_mode=target_type,
            color_mode=color_mode,
            shuffle=shuffle
        )

    def _create_train_val_generators(self,image_size,batch_size,target_type):
        """
        Create training and validation generators using the specified image_size, batch_size, and target_type.

        Args:
            image_size (tuple): Tuple of (height, width) for resizing images.
            batch_size (int): Number of image samples per batch.
            target_type (str): Type of target labels ('binary' or 'categorical').
        """
        train_val_data = self.kwargs['paths']['train_val_data']
        self.train_generator = self._create_generator(self.train_datagen,os.path.join(train_val_data,'train'),
                                                      image_size,batch_size,target_type)
        self.val_generator = self._create_generator(self.val_datagen,os.path.join(train_val_data,'val'),
                                                    image_size,batch_size,target_type)

    def _create_test_generators(self,image_size,batch_size,target_type):
        """
        Create test generators for internal and external test data using the specified image_size, batch_size, and target_type.

        Args:
            image_size (tuple): Tuple of (height, width) for resizing images.
            batch_size (int): Number of image samples per batch.
            target_type (str): Type of target labels ('binary' or 'categorical').
        """
        test_data_folder = self.kwargs['paths']['test_data_folder']
        if test_data_folder is not None:
            self.test_generator = self._create_generator(self.test_datagen,os.path.join(test_data_folder,'test'),
                                                         image_size,batch_size,target_type,shuffle=False)
        else:
            self.test_generator = None

        external_test_data_folder = self.kwargs['paths']['external_test_data_folder']
        if external_test_data_folder is not None:
            self.external_test_generator = self._create_generator(self.test_datagen,external_test_data_folder,
                                                                  image_size,batch_size,target_type,shuffle=False)
        else:
            self.external_test_generator = None

    def _create_all_generators(self):
        """
        Create all data generators (train, validation, test, and external test) based on the settings dictionary.
        """
        preprocess_input_func = self.preprocess_input
        image_size = self.kwargs['model']['image_size']
        batch_size = self.kwargs['training']['batch_size']
        target_type = self.kwargs['model']['target_type']

        self._create_datagens(preprocess_input_func)
        self._create_train_val_generators(image_size,batch_size,target_type)
        self._create_test_generators(image_size,batch_size,target_type)

    def _compute_class_weights(self):
        """
        Compute class weights for balanced training based on the unique target classes in the training data.
        """
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=np.unique(self.train_generator.classes),
                                             y=self.train_generator.classes)
        self.class_weights_dict = dict(zip(np.unique(self.train_generator.classes),class_weights))

    def _create_model(self):
        """
        Create and assign a Keras model to the instance, using the settings from the instance's `kwargs` attribute.
        """
        self.model = create_model(self.base_model,self.kwargs['model']['before_dense'],
                                  self.kwargs['model']['regularization'],
                                  self.kwargs['model']['number_of_targets'],
                                  self.kwargs['model']['dense_layers'],
                                  self.kwargs['model']['initializer'],self.kwargs['model']['batch_norm'],
                                  self.kwargs['model']['l1_strength'],self.kwargs['model']['l2_strength'],
                                  self.kwargs['model']['dropout_rate'],self.kwargs['model']['dense_activations'],)

    def _compile_model(self):
        """
        Compile the model with the specified optimizer, loss function, and metrics from the instance's `kwargs` attribute.
        """
        optimizer = self._get_optimizer(self.kwargs['training']['optimizer_name'],
                                        self.kwargs['training']['learning_rate'],
                                        self.kwargs['training']['add_optimizer_params'])
        loss = self._get_loss()
        metrics = self._get_metrics(self.kwargs['training']['metrics'])
        self.model.compile(optimizer=optimizer,loss=loss,metrics=metrics)

    def _load_user_preloaded_weights(self):
        """
        Load preloaded model weights from a file specified in the instance's `kwargs` attribute.
        """
        if self.kwargs['evaluation']['preloaded_weights_path'] is not None:
            self.model.load_weights(self.kwargs['evaluation']['preloaded_weights_path'])
            print(f"Loaded preloaded weights from {self.kwargs['evaluation']['preloaded_weights_path']}")

    def _train_main_model(self):
        """
        Train the model using the training generator, validation generator, and other settings from the instance's `kwargs` attribute.

        Returns:
            history (tensorflow.python.keras.callbacks.History): A Keras History object containing the training history.
        """

        history = self.model.fit(self.train_generator,
                                 epochs=self.kwargs['training']['epochs'],
                                 validation_data=self.val_generator,
                                 callbacks=self.callback_list,verbose=True,
                                 class_weight=self.class_weights_dict if self.kwargs[
                                     'training']['class_weights'] else None)
        return history

    def model_predict(self, path_to_folder, batch_size=32, verbose=True, sort_by='variance'):
        """
        Predict the classes of images in a folder using the trained model.

        Args:
            path_to_folder (str): Path to the directory containing images.
            batch_size (int, optional): Number of image samples per batch. Defaults to 32.
            verbose (bool, optional): Whether to print results during prediction. Defaults to True.
            sort_by (str, optional): Method to sort the predicted results. Defaults to 'variance'.

        Returns:
            results (list): A list of  dictionaries containing the prediction results.
        """
        results = model_predict(self.model, self.preprocess_input, path_to_folder,
                                self._class_labels, self.kwargs['model']['image_size'],
                                self.kwargs['model']['target_type'], batch_size, verbose, sort_by)
        return results

    def _display_summary(self):
        """
        Display the model summary if `show_summary` is True in the instance's `kwargs` attribute.
        """
        if self.kwargs['misc']['show_summary']:
            print("Model Summary:")
            self.model.summary()

    def _get_and_print_test_results(self, test_generator, results, label):
        """
        Evaluate the model using the provided test_generator, store and print the results.

        Args:
            test_generator (ImageDataGenerator): A data generator to provide test samples.
            results (dict): A dictionary to store test results.
            label (str): A label to identify the results (e.g. 'test' or 'external_test').
        """
        test_results = self.model.evaluate(test_generator, verbose=1)
        test_loss, test_metric = test_results[0], test_results[1]
        user_metric = self.kwargs['training']['metrics'][0].lower()
        results[f'{label}_loss'] = test_loss
        results[f'{label}_{user_metric}'] = test_metric

        print(f'{label.capitalize()} Loss: {test_loss:.4f}')
        print(f'{label.capitalize()} {user_metric.capitalize()}: {test_metric:.4f}')

        return test_metric

    def _process_test_results(self):
        """
        Process and plot the test results for both the test_generator and the external_test_generator.

        Returns:
            results (dict): A dictionary containing the test results (including plot objects).
        """
        results = {}

        if self.test_generator is not None:
            self._get_and_print_test_results(self.test_generator,results,'test')
            if self.kwargs['misc']['plot_conf'] is not None:
                plot_helpers.plot_confusion_matrix(self.model,self.test_generator,"Test",
                                                 self.kwargs['model']['target_type'])

        if self.external_test_generator is not None:
            self._get_and_print_test_results(self.external_test_generator,results,'external_test')
            if self.kwargs['misc']['plot_conf'] is not None:
                plot_helpers.plot_confusion_matrix(self.model,self.external_test_generator,"External Test",
                                                 self.kwargs['model']['target_type'])

        return results

    def _plot_history_curves(self, results):
        """
        Plot the training history curves using the plot_helpers.plot_history_curves utility.

        Args:
            results (dict): A dictionary containing training results and configuration.
        """
        if self.kwargs['misc']['plot_curves']:
            plot_helpers.plot_history_curves(results['history'],
                                           self.kwargs['misc']['show_min_max_plot'],
                                           self.kwargs['training']['metrics'][0].lower())


    def _update_progress_bar(self,current,total):
        """
        Update and print the progress bar.

        Args:
            current (int): The current progress.
            total (int): The total progress to be made.
        """
        progress = int(current/total*100)
        print(f'\rProcessing generator batches: [{("="*progress) + (" "*(100 - progress))}] {progress}%',end='')
        sys.stdout.flush()


    def _get_layer_output(self,layer_index=-2,layer_name=None):
        """
        Get the output tensor of a specific layer in the model.
        Args:
            layer_index (int, optional): Index of the desired layer. Defaults to -2.
            layer_name (str, optional): Name of the desired layer. Defaults to None.

        Returns:
            layer (tensorflow.python.keras.engine.keras_tensor.KerasTensor): Output tensor of the specified layer.
        """
        if layer_name:
            layer = None
            for l in self.model.layers:
                if l.name == layer_name:
                    layer = l.output
                    break
            if layer is None:
                raise ValueError(f"Layer with name '{layer_name}' not found.")
        else:
            layer = self.model.layers[layer_index].output
        return layer

    def _create_feature_extractor(self,layer_output):
        """
        Create a feature extractor model using the specified layer output.

        Args:
            layer_output (tensorflow.python.keras.engine.keras_tensor.KerasTensor): Output tensor of a specific layer.

        Returns:
            feature_extractor (tensorflow.python.keras.engine.functional.Functional): A Keras model to extract features.
        """
        feature_extractor = tf.keras.Model(inputs=self.model.inputs,outputs=layer_output)
        return feature_extractor

    def _extract_features_and_labels(self,data_generator,feature_extractor):
        """
        Extract features and labels from a data generator using a feature extractor model.
        Args:
            data_generator (ImageDataGenerator): A data generator to provide input samples.
            feature_extractor (tensorflow.python.keras.engine.functional.Functional): A Keras model to extract features.

        Returns:
            features (np.ndarray): A NumPy array of extracted features.
            labels (np.ndarray): A NumPy array of corresponding class labels.
        """
        features,labels = [],[]
        num_batches = len(data_generator)
        for i in range(num_batches):
            batch_x,batch_y = data_generator[i]
            batch_features = feature_extractor.predict(batch_x)
            features.extend(batch_features)

            if self.kwargs['model']['target_type'] == 'binary':
                threshold = 0.5
                class_indices = (batch_y > threshold).astype(int)
            else:
                class_indices = np.argmax(batch_y,axis=-1)

            labels.extend(class_indices)
            self._update_progress_bar(i + 1,num_batches)
        print('\n')
        return np.array(features),np.array(labels)

    def model_feature_extract(self,layer_index=-2,layer_name=None):
        """
        Perform feature extraction for train, validation, test, and external test generators using the specified layer.

        Args:
            layer_index (int, optional): Index of the desired layer for feature extraction. Defaults to -2.
            layer_name (str, optional): Name of the desired layer for feature extraction. Defaults to None.

        Returns:
            X_train (np.ndarray): Extracted features for the training set.
            y_train (np.ndarray): Labels for the training set.
            X_val (np.ndarray): Extracted features for the validation set.
            y_val (np.ndarray): Labels for the validation set.
            X_test (np.ndarray): Extracted features for the test set, or None if no test generator is provided.
            y_test (np.ndarray): Labels for the test set, or None if no test generator is provided.
            X_test_external (np.ndarray): Extracted features for the external test set, or None if no external test generator is provided.
            y_test_external (np.ndarray): Labels for the external test set, or None if no external test generator is provided.
        """
        layer_output = self._get_layer_output(layer_index,layer_name)
        feature_extractor = self._create_feature_extractor(layer_output)
        X_train,y_train = self._extract_features_and_labels(self.train_generator,feature_extractor)
        X_val,y_val = self._extract_features_and_labels(self.val_generator,feature_extractor)

        if self.test_generator is not None:
            X_test,y_test = self._extract_features_and_labels(self.test_generator,feature_extractor)
        else:
            X_test,y_test = None,None

        if self.external_test_generator is not None:
            X_test_external,y_test_external = self._extract_features_and_labels(self.external_test_generator,
                                                                               feature_extractor)
        else:
            X_test_external,y_test_external = None,None

        return X_train,y_train,X_val,y_val,X_test,y_test,X_test_external,y_test_external

    def _warm_pretrain_dense(self):
        """
        Warm pretrain dense layers of the model while keeping the base transfer architecture layers frozen.
        """
        if self.kwargs['training']['warm_pretrain_dense']:
            for layer in self.base_model.layers:
                layer.trainable = False
            print(f"Warm pretraining dense layers (transfer architecture is frozen)...")
            warm_epochs = self.kwargs['training']['warm_pretrain_epochs']
            self.model.fit(self.train_generator,
                           epochs=warm_epochs,
                           validation_data=self.val_generator,
                           verbose=True,callbacks=self.callback_list,
                           class_weight=self.class_weights_dict if self.kwargs['training']['class_weights'] else None)
            self._reset_trainable()
            print ( f"Warm pretraining of dense layers completed (transfer architecture is now back to user configuration)." )

    def _train_or_evaluate(self):
        """
        Train the main model or evaluate it based on the 'evaluate_mode' flag.

        Returns:
            results (dict): A dictionary containing the history if the model was trained.
        """
        if not self.kwargs['evaluation']['evaluate_mode']:
            history = self._train_main_model()
            results = {'history': history}
        else:
            results = {}

        return results

    @property
    def _class_labels(self):
        """
        Get the class labels from the training generator.

        Returns:
            class_labels (list): A list of class labels.
        """
        return list(self.train_generator.class_indices.keys())

    def export_all(self,results,base_path="exports",export_model=True,additive=True):
        """
        Wrapper function to export the results, model, and configuration of the training and testing process.

        Args:
            results (dict): The results of the training and testing process.
            base_path (str, optional): The base path for saving the exported files. Defaults to "exports".
            export_model (bool, optional): Whether to save the model architecture and weights. Defaults to True.
            additive (bool, optional): Whether to create a new folder with a random name inside the base path. Defaults to True.
        """
        export(self.model,self.kwargs,results,base_path,export_model,additive)

    def _reset_trainable(self):
        """
        Reset the trainable status of all layers in the model to True.
        """
        for layer in self.model.layers:
            layer.trainable = True

    def _freeze_weights(self):
        """
        Freeze the weights of the first N layers of the transfer architecture as specified by the 'freeze_weights' parameter.
        """
        if self.kwargs['model']['freeze_weights'] != 0:
            for layer in self.base_model.layers[:self.kwargs['model']['freeze_weights']]:
                layer.trainable = False

    def _unfreeze_block(self):
        """
        Unfreeze the specified block of layers in the model.
        """
        if self.kwargs['model']['unfreeze_block']:
            for layer in self.model.layers:
                if any([layer.name.startswith(block_name) for block_name in self.kwargs['model']['unfreeze_block']]):
                    layer.trainable = True

    def _freeze_up_to_block(self):
        """
        Freeze layers up to the specified block in the model.
        """
        freeze_block = self.kwargs['model']['freeze_up_to']
        if freeze_block:
            freeze_reached = False
            for layer in self.model.layers:
                if not freeze_reached:
                    layer.trainable = False
                if layer.name.startswith(freeze_block):
                    freeze_reached = True
                if freeze_reached:
                    layer.trainable = True
            print(f"Freezing up to block {freeze_block}.")

    def _show_freeze_status(self):
        """
        Print the trainable status of each layer in the model.
        """
        if self.kwargs['model']['show_freeze_status']:
            print("Layer Trainable Status:")
            for layer in self.model.layers:
                print(f"Layer: {layer.name}, Trainable: {layer.trainable}")

    def _prepare_callbacks(self):
        """
        Prepare the list of Keras callbacks for model training.
        """
        self.callback_list = []
        if self.kwargs['training']['callback']:
            for i in self.kwargs['training']['callback']:
                self.callback_list.append(i)

        if self.kwargs['training']['early_stop']:
            early_stopping_callback = EarlyStopping(monitor='val_loss',
                                                    patience=int(self.kwargs['training']['epochs']*
                                                                 self.kwargs['training']['early_stop']))
            self.callback_list.append(early_stopping_callback)

        if self.kwargs['saving']['save_weights']:
            custom_save_weights_callback = CustomSaveWeights(self.kwargs['saving']['save_weights_folder'],
                                                             self.kwargs['saving']['save_best_weights'],
                                                             save_weights_silent=self.kwargs['saving'][
                                                                 'save_weights_silent'])
            self.callback_list.append(custom_save_weights_callback)

    def _initialize(self):
        """
        Initialize the pipeline by setting the number of targets, target type,
        transfer architecture, generators, class weights, and callbacks.
        Also compiles the model and loads preloaded weights if provided.
        """
        self._set_number_of_targets()
        self._set_target_type()
        self._initialize_base_model_and_fetch_prepro()
        self._create_all_generators()
        self._compute_class_weights()
        self._create_model()
        self._compile_model()
        self._load_user_preloaded_weights()
        self._prepare_callbacks()
        self._warm_pretrain_dense()
        self._freeze_weights()
        self._freeze_up_to_block()
        self._unfreeze_block()
        self._display_summary()
        self._show_freeze_status()

    def run(self):
        """
        Run the main pipeline, including initialization, training or evaluating, and processing the test results.

        Returns:
            model (tensorflow.python.keras.engine.functional.Functional): The trained or evaluated Keras model.
            results (dict): A dictionary containing training results and configuration.
        """
        self._initialize()
        results = self._train_or_evaluate()
        self._plot_history_curves(results)
        results['configuration'] = self.export_configuration()
        self._auto_evaluate()
        results.update(self._process_test_results())

        return self.model, results