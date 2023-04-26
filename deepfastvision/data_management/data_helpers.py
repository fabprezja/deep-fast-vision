# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import os
import shutil
import numpy as np
import sys

def update_nested_dict(d, u):
    """
    Updates a nested dictionary with values from another dictionary.

    Args:
        d (dict): The dictionary to update.
        u (dict): The dictionary to use for updating.

    Returns:
        dict: The updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class DatasetSplitter:
    """
    DatasetSplitter class for splitting a dataset into train, validation, and test sets.

    Attributes:
        data_dir (str): Directory containing the dataset.
        destination_dir (str): Directory where the train, validation, and test sets will be saved.
        train_ratio (float): Ratio of train set.
        val_ratio (float): Ratio of validation set.
        test_ratio (float): Ratio of test set.
        test_ratio_2 (float): Ratio of an additional test set, if needed.
        seed (int): Seed for random number generator.
    """
    def __init__(self, data_dir, destination_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, test_ratio_2=None,
                 seed=None):
        self.data_dir = data_dir
        self.destination_dir = destination_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.test_ratio_2 = test_ratio_2
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def run(self):
        """
        Main method to execute dataset splitting and copying files.
        """
        if not os.path.exists(self.destination_dir):
            os.makedirs(self.destination_dir)

        subdirs = self._get_subdirs()
        total_files = self._count_total_files(subdirs)

        processed_files = 0
        for subdir in subdirs:
            subdir_path = os.path.join(self.data_dir, subdir)
            files = os.listdir(subdir_path)
            np.random.shuffle(files)

            train_split, val_split, test_split, test_split_2 = self._calculate_split_indices(len(files))

            subdir_processed_files = self._copy_files(files, subdir, train_split, val_split, test_split,
                                                      test_split_2)
            processed_files += subdir_processed_files
            self._update_progress_bar(processed_files, total_files)

    def _get_subdirs(self):
        """
        Gets the list of subdirectories in the data directory.

        Returns:
            list: List of subdirectories.
        """
        return [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

    def _count_total_files(self, subdirs):
        """
        Counts the total number of files in the given subdirectories.

        Args:
            subdirs (list): List of subdirectories.

        Returns:
            int: Total number of files.
        """
        return sum(len(os.listdir(os.path.join(self.data_dir, subdir))) for subdir in subdirs)

    def _calculate_split_indices(self, num_files):
        """
        Calculates the split indices for train, validation, test, and additional test set (if applicable).

        Args:
            num_files (int): Total number of files in the subdirectory.

        Returns:
            tuple: Train split index, validation split index, test split index, and additional test split index (if applicable).
        """
        train_split = int(self.train_ratio * num_files)
        val_split = int((self.train_ratio + self.val_ratio) * num_files)
        test_split = int((self.train_ratio + self.val_ratio + self.test_ratio) * num_files) if self.test_ratio else None
        test_split_2 = int((self.train_ratio + self.val_ratio + self.test_ratio + self.test_ratio_2) * num_files) if self.test_ratio_2 else None
        return train_split, val_split, test_split, test_split_2

    def _create_destination_path(self, file_index, subdir, train_split, val_split, test_split):
        """
        Creates the destination path for a file based on its index.

        Args:
            file_index (int): Index of the file in the subdirectory.
            subdir (str): Subdirectory name.
            train_split (int): Train split index.
            val_split (int): Validation split index.
            test_split (int): Test split index.

        Returns:
            str: Destination path for the file.
        """
        if file_index < train_split:
            return os.path.join(self.destination_dir, 'train', subdir)
        elif file_index < val_split:
            return os.path.join(self.destination_dir, 'val', subdir)
        elif self.test_ratio and file_index < test_split:
            return os.path.join(self.destination_dir, 'test', subdir)
        elif self.test_ratio_2:
            return os.path.join(self.destination_dir, 'test_set_2', subdir)
        else:
            return None

    def _copy_files(self, files, subdir, train_split, val_split, test_split, test_split_2):
        """
        Copies files from the source directory to the destination directory based on the split indices.

        Args:
            files (list): List of file names.
            subdir (str): Subdirectory name.
            train_split (int): Train split index.
            val_split (int): Validation split index.
            test_split (int): Test split index.
            test_split_2 (int): Second test set split index, if applicable.

        Returns:
            int: Number of processed files.
        """
        processed_files = 0

        for i, file in enumerate(files):
            src_path = os.path.join(self.data_dir, subdir, file)
            dest_path = self._create_destination_path(i, subdir, train_split, val_split, test_split)

            if dest_path is None:
                continue

            os.makedirs(dest_path, exist_ok=True)
            shutil.copy(src_path, os.path.join(dest_path, file))
            processed_files += 1

        return processed_files

    def _update_progress_bar(self,current,total):
        """
        Updates the progress bar based on the current number of processed files and the total number of files.

        Args:
            current (int): Current number of processed files.
            total (int): Total number of files.
        """
        progress = int((current/total)*100)
        bar = '='*progress + ' '*(100 - progress)
        print(f'\rSplitting Data: [{bar}] {progress}%',end='',flush=True)

class DataSubSampler:
    """
    DataSubSampler class for creating a smaller dataset by randomly sampling a fraction of files from the original dataset.

    Attributes:
        data_dir (str): Directory containing the original dataset.
        destination_dir (str): Directory where the sampled dataset will be saved.
        fraction (float): Fraction of files to sample from the original dataset.
        seed (int): Seed for random number generator.
    """
    def __init__(self, data_dir, destination_dir, fraction, seed=None):
        self.data_dir = data_dir
        self.destination_dir = destination_dir
        self.fraction = fraction
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    def create_miniature_dataset(self):
        """
        Creates a copy of all folders and subfolders in the path, but only samples a fraction of files.
        """
        for root, dirs, files in os.walk(self.data_dir):
            destination_root = root.replace(self.data_dir, self.destination_dir)

            if not os.path.exists(destination_root):
                os.makedirs(destination_root)

            files_to_sample = int(len(files) * self.fraction)
            sampled_files = np.random.choice(files, files_to_sample, replace=False)

            for file in sampled_files:
                src_path = os.path.join(root, file)
                dest_path = os.path.join(destination_root, file)
                shutil.copy(src_path, dest_path)