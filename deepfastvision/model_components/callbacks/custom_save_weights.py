# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import os
from tensorflow.keras.callbacks import Callback

class CustomSaveWeights(Callback):
    """
    Custom Keras callback to save model weights during training based on a specified metric.

    Attributes:
        weights_folder (str): Path to the directory where weights files will be saved.
        save_weights_mode (str): Mode of saving weights, either 'all', 'val_loss', or any other metric name.
        save_weights_silent (bool, optional): If True, suppresses the print statements during weight saving. Defaults to False.
        base_val_metric (float): The base validation metric achieved so far during training. Initialized based on the provided metric.
    """
    def __init__(self, weights_folder, save_weights_mode, save_weights_silent=False):
        """
        Initializes the CustomSaveWeights callback with the provided arguments.

        Args:
            weights_folder (str): Path to the directory where weights files will be saved.
            save_weights_mode (str): Mode of saving weights, either 'all', 'val_loss', or any other metric name.
            save_weights_silent (bool, optional): If True, suppresses the print statements during weight saving. Defaults to False.
        """
        super(CustomSaveWeights, self).__init__()
        self.weights_folder = weights_folder
        self.save_weights_mode = save_weights_mode
        self.save_weights_silent = save_weights_silent
        self.base_val_metric = 0.0 if save_weights_mode != 'val_loss' else float('inf')

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch. Saves the model weights based on the provided mode.

        Args:
            epoch (int): The current epoch number.
            logs (dict, optional): Dictionary of logs containing training and validation metrics. Defaults to None.
        """
        if self.save_weights_mode == 'all':
            save_path = os.path.join(self.weights_folder, f'weights_epoch_{epoch + 1}.h5')
            self.model.save_weights(save_path)
            if not self.save_weights_silent:
                print(f"Saved weights for epoch {epoch + 1} at {save_path}")
        else:
            current_val_metric = logs.get(self.save_weights_mode)
            should_save = (self.save_weights_mode == 'val_loss' and current_val_metric < self.base_val_metric) or \
                          (self.save_weights_mode != 'val_loss' and current_val_metric > self.base_val_metric)

            if should_save:
                self.base_val_metric = current_val_metric
                save_path = os.path.join(self.weights_folder, f'best_weights_epoch_{epoch + 1}.h5')
                self.model.save_weights(save_path)
                if not self.save_weights_silent:
                    print(f"Saved best weights for epoch {epoch + 1} at {save_path}")
