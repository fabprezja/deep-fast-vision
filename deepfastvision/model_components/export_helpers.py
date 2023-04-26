# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import uuid
import os
import pandas as pd
import pprint


def export(model, kwargs, results, base_path="exports", export_model=False, additive=True):
    """
    Exports the results of the training and testing process.

    Args:
        model (Model): The trained model.
        kwargs (dict): The configuration used for training.
        results (dict): The pipeline results of the training and testing process.
        base_path (str, optional): The base path for saving the exported files. Defaults to "exports".
        export_model (bool, optional): Whether to save the model architecture and weights. Defaults to False.
        additive (bool, optional): Whether to generate a random folder inside the base path. If set to False,
        the export happens exactly at the user-specified folder. Defaults to True.
    """
    if additive:
        folder_name = str(uuid.uuid4())
        export_path = os.path.join(base_path, folder_name)
    else:
        export_path = base_path

    os.makedirs(export_path, exist_ok=True)

    export_history_to_csv(results['history'], os.path.join(export_path, 'training_history.csv'))
    export_kwargs(kwargs, os.path.join(export_path, 'kwargs.txt'))
    export_test_results(results, os.path.join(export_path, 'test_results.csv'), kwargs)
    export_best_weights(model, os.path.join(export_path, 'best_weights.h5'))

    if export_model:
        model.save(os.path.join(export_path, 'model.h5'))

    print(f"Exported results to {export_path}")

def export_best_weights(model, file_path):
    """
    Exports the best weights of the model.

    Args:
        model (Model): The trained model.
        file_path (str): The path where the best weights will be saved.
    """
    model.save_weights(file_path)

def export_test_loss(test_loss,file_path):
    """
    Exports the test loss.

    Args:
        test_loss (float): The test loss value.
        file_path (str): The path where the test loss will be saved.
    """
    with open(file_path,"w") as f:
        f.write(f"Test Loss: {test_loss}")

def export_history_to_csv(history, file_path):
    """
    Exports the training history to a CSV file.

    Args:
        history (History): The training history object.
        file_path (str): The path where the history CSV file will be saved.
    """
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(file_path, index=False)

def export_kwargs(kwargs, file_path):
    """
    Exports the training configuration (kwargs) to a file.

    Args:
        kwargs (dict): The configuration used for training.
        file_path (str): The path where the configuration will be saved.
    """
    with open(file_path, "w") as f:
        pprint.pprint(kwargs, stream=f, indent=4, width=100)

def export_test_results(results, file_path, kwargs):
    """
    Exports the test results to a CSV file.

    Args:
        results (dict): The pipeline results of the training and testing process.
        file_path (str): The path where the test results CSV file will be saved.
        kwargs (dict): The configuration used for training.
    """
    test_loss = results['test_loss']
    first_metric = kwargs['training']['metrics'][0].lower()
    first_metric_value = results[f'test_{first_metric}']
    external_test_loss = results.get('external_test_loss', None)
    external_first_metric_value = results.get(f'external_test_{first_metric}', None)

    test_data = {
        'Test Loss': [test_loss],
        f'Test {first_metric.capitalize()}': [first_metric_value]
    }

    if external_test_loss is not None:
        test_data['External Test Loss'] = [external_test_loss]

    if external_first_metric_value is not None:
        test_data[f'External Test {first_metric.capitalize()}'] = [external_first_metric_value]

    test_results_df = pd.DataFrame(test_data)
    test_results_df.to_csv(file_path, index=False)

