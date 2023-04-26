# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, generator, generator_name, target_type):
    """
    Plot the confusion matrix of the model predictions on a given dataset.

    Args:
        model (Model): A trained Keras model.
        generator (ImageDataGenerator): A Keras ImageDataGenerator.
        generator_name (str): Name of the generator, used in the plot title.
        target_type (str): Type of target labels ('binary' or 'categorical').
    """
    true_labels = generator.classes
    pred_prob = model.predict(generator)

    if target_type == 'binary':
        threshold = 0.5
        pred_labels = (pred_prob > threshold).astype(int)
    else:
        pred_labels = np.argmax(pred_prob, axis=1)

    cm = confusion_matrix(true_labels, pred_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_labels = list(generator.class_indices.keys())
    plt.figure()
    sns.heatmap(cm_norm, annot=True, cmap="Blues", fmt='.2f', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'{generator_name} Normalized Confusion Matrix')
    plt.show()


def plot_history_curves(history, show_min_max_plot, user_metric):
    """
    Plot the training and validation loss and user-specified metric curves.

    Args:
        history (History): Keras training history object.
        show_min_max_plot (bool): Whether to plot the maximum/minimum value lines.
        user_metric (str): User-specified metric to be plotted.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    plot_curve(ax1, history, 'loss', 'Loss', show_min_max_plot)
    train_metric_key = next(key for key in history.history.keys() if key.lower().startswith(user_metric))
    val_metric_key = next(key for key in history.history.keys() if key.lower().startswith(f'val_{user_metric}'))

    plot_curve(ax2, history, train_metric_key, user_metric.capitalize(), show_min_max_plot)

    plt.tight_layout()
    plt.show()


def plot_curve(ax, history, metric, ylabel, show_min_max_plot):
    """
    Plot a curve of a specific metric.

    Args:
        ax (Axes): Matplotlib Axes object.
        history (History): Keras training history object.
        metric (str): Metric to be plotted.
        ylabel (str): Label for the y-axis.
        show_min_max_plot (bool): Whether to plot the maximum/minimum value lines.
    """
    ax.plot(history.history[metric], label=f'Train {ylabel}')
    ax.plot(history.history[f'val_{metric}'], label=f'Validation {ylabel}')

    if show_min_max_plot:
        plot_max_min_lines(ax, history, metric, ylabel)

    set_plot_labels(ax, history, metric, ylabel)

    ax.legend()
    ax.set_xticks(range(0, len(history.history[metric])))
    ax.set_xticklabels([str(x + 1) for x in ax.get_xticks()])


def plot_max_min_lines(ax, history, metric, ylabel):
    """
    Plot the maximum/minimum value lines for a specific metric.

    Args:
        ax (Axes): Matplotlib Axes object.
        history (History): Keras training history object.
        metric (str): Metric to be plotted.
        ylabel (str): Label for the y-axis.
    """
    val_metric_key = f'val_{metric}'
    if val_metric_key in history.history:
        val_metric_epoch = np.argmax(history.history[val_metric_key]) if ylabel != 'Loss' \
            else np.argmin(history.history[val_metric_key])
        max_val_metric = np.max(history.history[val_metric_key]) if ylabel != 'Loss' \
            else np.min(history.history[val_metric_key])
        ax.axvline(x=val_metric_epoch, color='g', linestyle='--', alpha=0.5,
                   label=(f'Maximum Validation {ylabel}: {max_val_metric:.4f}' if ylabel != 'Loss'
                          else f'Minimum Validation Loss: {max_val_metric:.4f}'))


def set_plot_labels(ax, history, metric, ylabel):
    """
    Set the plot labels, grid, legend, and tick labels for a specific metric.

    Args:
        ax (Axes): Matplotlib Axes object.
        history (History): Keras training history object.
        metric (str): Metric to be plotted.
        ylabel (str): Label for the y-axis.
    """
    ax.set_title(f'{ylabel} Curves')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.grid(True,linestyle='--',alpha=0.5)
    ax.legend()
    ax.set_xticks(range(0,len(history.history[metric])))
    ax.set_xticklabels([str(x + 1) for x in ax.get_xticks()])