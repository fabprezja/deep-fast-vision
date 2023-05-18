# Deep Fast Vision: Accelerated Deep Transfer Learning Vision Prototyping and Beyond
<img src="https://user-images.githubusercontent.com/87379098/234222583-5f1fcbf6-368d-471d-8f64-fa25b6ccf925.png" alt="logo" width="60%">

Deep Fast Vision is a versatile Python library for rapid prototyping of deep transfer learning vision models. It caters to users of various levels, offering different levels of abstraction from high-level configurations for beginners to mid and low-level customization for professional data scientists and developers. Built around Keras and TensorFlow, this library also includes handy utilities.

Compute mode depends on Tensorflow configuration: GPU or CPU (GPU is recommended).

## (a few) Key Features

1. Auto loss/target type determination
2. Auto generator setup
3. Auto output layer setup
4. Auto pre-training of new dense layers before unfreezing transfer architecture (in parts or as a whole)
5. Auto augmentation setup (from templates and/or Custom)
6. Auto best weights saving and loading
7. Auto class weights calculation
8. Auto validation curves plot (with minimúm loss & maximum metric epoch highlight)
9. Auto confusion matrices for test/external data
10. Easy dense layer configuration
11. Easy regularization set up and mixing (Dropout, L2, L1, Early Stop, etc.)
12. Access to all Keras optimizers & callback support

Comprehensive documentation for Deep Fast Vision is available both in the docs folder and at the [**documentation page**](https://fabprezja.github.io/deep-fast-vision/).


## Install using pip:

You can install `deepfastvision` using pip with the following command:

```shell
pip install deepfastvision
```
You can install `deepfastvision` with the older `tensorflow-gpu` using the following command:

```shell
pip install deepfastvision[gpu]
```
## How to cite:

If you find the work usefull in your project, please cite:

```bibtex
@misc{fabprezja_2023,
  author = {Fabi Prezja},
  title = {Deep Fast Vision: Accelerated Deep Transfer Learning Vision Prototyping and Beyond},
  month = apr,
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/fabprezja/deep-fast-vision}},
  doi = {10.5281/zenodo.7865289},
  url = {https://doi.org/10.5281/zenodo.7865289}
}
```
# Usage Examples by Abstraction Level (and Capabilitity details)
Bellow are examples by level of abstraction while detailing automation and configuration capabilities.

## High Level of Abstraction
```python
import wandb
from wandb.keras import WandbCallback
from deepfastvision.core import DeepTransferClassification

# Initialize wandb
wandb.init(project='your_project_name', entity='your_username')

# Create a DeepTransferClassification object
experiment = DeepTransferClassification(paths={'train_val_data': 'path_to_train_val_data',
                                               'test_data_folder': 'path_to_test_data'},
                                        saving={'save_weights_folder':'path_to_save_weights'},
                                        model= {'transfer_arch': 'VGG16',
                                                'dense_layers': [144,89,55],
                                                'unfreeze_block': ['block5']},
                                        training={'epochs': 15,
                                                  'learning_rate': 0.0001,
                                                  'metrics': ['accuracy'],
                                                  'callback': [WanDBCallback()]})

model, results = experiment.run()
```

The above code will return:

-   **Trained model**: The final model after training.
-   **Results dictionary**: A results dictionary which contains evaluation and training results, as well as model and training configuration.
-   **Validation curves**: Automatically configured based on the target type and provided labels & metrics.
-   **Confusion matrix**: Automatically configured based on the target type and provided labels & metrics.

**User provided:**
1. Data paths: The library identifies and loads the data from the provided paths for training, validation, and testing.
2. Transfer learning architecture (VGG16): The library fetches the VGG16 pre-trained model and uses it as the base for the new neural network.
3. Dense layer configuration [144, 89, 55]: The library creates dense layers with 144, 89, and 55 neurons, respectively, and adds them to the neural network.
4. Any callback: Insert the user-defined callback (WanDB in this case).

**After processing the user-provided information, the library automatically performs the following tasks to create, train, and evaluate the neural network model (customizable in lower levels of abstraction):**

1. Determine Loss function
2. Identify Train-Val and Test folders in the provided path
3. Establish output layer size and activation functions
4. Calculate and apply class weights (adjustable)
5. Insert dropout between the dense layers (adjustable).
6. Generate appropriate data generators for train, val, and test data
7. Resize all images to 224 x 224 which i.e, the transfer model specification (adjustable).
8. Retrieve transfer architecture's preprocessing function for augmentation settings
9. Prepare data augmentation for training data generator (adjustable)
10. Pre-train dense layers with a frozen transfer model, then unfreeze and train as specified (to mitigate destructive effects on unfrozen transfer architecture)  (adjustable)
11. Monitor and load optimal weights based on validation results before testing (adjustable)
12. Conduct test after loading the best weights (adjustable)
13. Create validation curves highlighting the best metric value and validation epoch (adjustable)
14. Produce confusion matrices for test set(s) (adjustable)
15. Provide model architecture summary and trainable layers summary (adjustable)

## Random Run Example
![image](https://user-images.githubusercontent.com/87379098/233811708-dea3958d-6439-424f-bdac-9af669da769c.png)

## Medium Level of Abstraction
```python
from deepfastvision.core import DeepTransferClassification

experiment = DeepTransferClassification(paths={'train_val_data': 'path_to_train_val_data',
                                               'test_data_folder': 'path_to_test_data'},
                                        saving={'save_weights_folder':'path_to_save_weights'},
                                        model= {'transfer_arch': 'VGG19',
                                                'dense_layers': [ 377, 233, 144, 89 ],
                                                'dense_activations': 'relu' ,
                                                'regularization': 'Dropout+L1',
                                                'dropout_rate': 0.35 ,
                                                'unfreeze_block': ['block5']},
                                        training={'epochs': 15,
                                                  'batch_size': 32 ,
                                                  'optimizer_name': 'Adam' ,
                                                  'add_optimizer_params': {'clipnorm': 0.8} ,
                                                  'augmentation': 'advanced' ,
                                                  'learning_rate': 0.0001,
                                                  'metrics': [ 'accuracy' , 'recall', 'precision']})

model, results = experiment.run()
```
In the medium level of abstraction example, the user provides more specific configurations for the model and training process in addition to the detailed automations, as compared to the high level of abstraction example. Here are the main differences:

1. **Transfer learning architecture**: This example selects VGG19 instead of VGG16.
1. **Dense layer configuration**: This example defines another dense layer configuration, specifying 377, 233, 144, and 89 neurons in respective layers.
2. **Dense activation function**: This example provides a specific activation function, 'ReLU', for the dense layers.
3. **Regularization**: This example chooses 'Dropout + L1 as for dense layer regularization.
4. **Dropout rate**: This example sets a custom dropout rate of 0.35.
5. **Training parameters**: This example provides more specific training parameters, including batch size (32), optimizer name ('Adam'), and additional optimizer parameters such as gradient cliping ({'clipnorm': 0.8}).
6. **Data augmentation**: This example selects an advanced data augmentation strategy (instead of the default 'basic').
7. **Metrics**: This example specifies additional metrics for evaluation, including recall and precision, along with accuracy. The first metric in the list is used as the metric sub-plot in validation curves.

These changes give more control over the model architecture and training process, allowing for more tailored experiments and better performance tuning. All previous detailed automations still apply.

## Low Level of Abstraction & Default Configuration
```python
from deepfastvision.core import DeepTransferClassification

experiment = DeepTransferClassification(paths={
    'train_val_data': 'path_to_train_val_data',
    'test_data_folder': 'path_to_test_data',
    'external_test_data_folder': 'path_to_external_test_data',
},
model={
    'image_size': (224, 224),
    'transfer_arch': 'VGG19',
    'pre_trained': 'imagenet',
    'before_dense': 'Flatten',
    'dense_layers': [610, 377, 233, 144, 89, 55],
    'dense_activations': 'elu',
    'initializer': 'he_normal',
    'batch_norm': True,
    'regularization': 'Dropout+L2',
    'l2_strength': 0.001,
    'dropout_rate': 0.35,
    'unfreeze_block': ['block1', 'block2', 'block5'],
    'freeze_up_to': 'flatten',
},
training={
    'epochs': 9,
    'batch_size': 32,
    'learning_rate': 2e-5,
    'optimizer_name': 'Adam',
    'add_optimizer_params': {'clipnorm': 0.8},
    'class_weights': True,
    'metrics': ['accuracy', 'recall', 'precision'],
    'augmentation': 'custom',
    'custom_augmentation':[user_function]
    'callback': [WandbCallback(), learning_rate_schedule],
    'early_stop': 0.20,
    'warm_pretrain_dense': True,
    'warm_pretrain_epochs': 9,
},
evaluation={
    'auto_mode': True,
},
saving={
    'save_weights': True,
    'save_weights_folder': 'path_to_save_weights',
    'save_best_weights': 'val_loss',
},
misc={
    'show_summary': True,
    'plot_curves': True,
    'show_min_max_plot': True,
    'plot_conf': True,
})

model, results = experiment.run()
```

In this low-level of abstraction example, the user has greater control over the model and training process, providing a more detailed configuration. This example includes new user changes and options, as well as the default values in the configuration. Compared to the previous example, the user can additionally specify:

Changes compared to the Medium level of abstraction:
1. **Paths:** The user added an 'external_test_data_folder' for additional test data.
2. **Batch normalization** was enabled.
3. **More Dense Layers** provided (610, 377, 233, 144, 89, 55)
4. **Regularization** was set to 'Dropout+L2', with L2 strength specified as 0.001.
5. The user specified which **blocks to unfreeze** ('block1', 'block2', 'block5') and froze layers up to 'flatten'.
6. **Augmentation** was set to 'custom' and provided by user.
7. **User-provided callbacks** were added: WandbCallback and learning_rate_schedule.
8. **Early stopping** tolerance was enabled with a threshold of 0.20 (as a ratio of total epochs).
9. **Epochs** were increased to 25.
10. **Warm pretraining** for dense layers was enabled with 9 epochs.

Actiavated by default:
1. **Weight initializer** is set to 'he_normal'.
2. **Class weights** are enabled.
3. **Image size** is explicitly set to (224, 224).
4. **Pre-trained weights** are specified to be from 'imagenet'.
5. **Layer before dense layers** is set to 'Flatten'.
6. **Dense layer activations** are set to 'elu'.
7. **Learning rate** is set to 2e-5.
8. **Evaluation configuration**: 'auto_mode' is True. (automatic evaluation of best weights)
9. **Saving configuration:** Saving best weights based on 'val_loss'.
10. **Miscellaneous settings:** Showing model summary, plotting curves, showing min-max plot, and plotting confusion matrix.

As in the previous examples all automations are applied.

### Loading Model Checkpoint and/or Inference Only
When evaluate_mode is True, training cannot occur. Similarly, when a preloaded weights path is given, the model initializes with the preloaded weights and allows further training or inference.

```python
saving={
    'evaluate_mode': True,
    'auto_mode': True,
    'preloaded_weights_path': 'path_to_preloaded_weights',
}
```

# Available Class Methods

**1. Model Prediction**

```python
predictions = experiment.model_predict('folder_path')
```
The model_predict method uses the trained model to predict all images in a given folder. The method returns image, path, predicted label, confidence, and variance for each image in the folder. It can be sorted by variance (across labels) for identifying confusing instances or by metric (e.g., accuracy).

**2. Export Results and Model**
```python
experiment.export_all(results, base_path='folder_path_to_results', export_model=True, additive=True)
```
The export_all method exports all results, best weights, and the trained model into a folder. With additive=True, the user may iterate the experiment and obtain results in new randomly named folders.

**3. Extract Features**
```python
X_train, y_train, X_val, y_val, X_test, y_test, X_test_external, y_test_external = experiment.model_feature_extract(layer_index=None, layer_name='block5')
```
The model_feature_extract method can be used to extract features from any layer in the model while respecting the used train, val, test(s) indices.

# Extra Utilities

**2. Data Splitter**
A class to split any data into the required partition format (train, val, test(s)). The splits are stratified.
```python
from deepfastvision.data_management.data_helpers import DatasetSplitter

# Define the paths to the original dataset and the destination directory for the split datasets
data_dir = 'path/to/original/dataset'
destination_dir = 'path/to/destination/directory'

# Instantiate the DatasetSplitter class with the desired train, validation, and test set ratios
splitter = DatasetSplitter(data_dir, destination_dir, train_ratio=0.7,
                           val_ratio=0.10, test_ratio=0.10, test_ratio_2=0.10, seed=42)

# Split the dataset into train, validation, and test sets
splitter.run()
```
**2. Data Sub Sampler (miniaturize)**
A class to sub-sample (miniaturize) any dataset give a ratio:
```python
from deepfastvision.data_management.data_helpers import DataSubSampler

# Define the paths to the original dataset and the destination directory for the subsampled dataset
data_dir = 'path/to/original/dataset'
subsampled_destination_dir = 'path/to/subsampled/dataset'

# Instantiate the DataSubSampler class with the desired fraction of files to sample
subsampler = DataSubSampler(data_dir, subsampled_destination_dir, fraction=0.5, seed=42)

# Create a smaller dataset by randomly sampling a fraction (in this case, 50%) of files from the original dataset
subsampler.create_miniature_dataset()
```
