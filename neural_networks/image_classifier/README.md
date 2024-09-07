# Image Classification Project

This project is the final project for this module, Neural Networks with Tensorflow.

The first part of the project is snapshotting the data, training a model based on that data, and finally predicting images using the trained network and achieving an accuracy of `>70%` with the test data. The trained model is saved in `HDF5` format (look for a file with `.h5` extension within this directory, like for example, `1725545374.h5`).

To achieve the same results, install the dependencies mentioned in `environment.yml` file. This file is an extract from the local virtual environment by using `$ conda env export > environment.yml`.

## Project Image Classifier Project Notebook

This notebook is where the training, validation, and testing data is snapshotted from `tensorflow datasets`. The [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) is the dataset used. The final model architecture used for this project is composed of the pretrained `Mobile Net` neural network cobined with a dense `Softmax` output layer. The `Mobile Net` network is used for transfer learning without and further training on the network. Only the added dense layer is trained in this process with an `epoch` of 5. The pretraining accuracy of the model was `1.56%` and post training, it was at `74.92%`. The trained model is saved as a Keras model (`HDF5` file with `.h5` extension). The notebook finally uses the model to perform inferences (with graphs and images) on sample images.

## Prediction

Though prediction/inference was done on the notebook, rerunning it causes model retraining. Instead, a standalone Python script, `predict.py` is provided for users to test and use the trained-and-saved model from the notebook. To use this script,

1. Make sure to install the required dependencies from `environment.yml`.
2. Prepare images to test with
3. Make sure to have saved Keras model (trained model) like for example, `1725545374.h5`.
4. {Optional} Have a JSON mapping of the flower names with their indexes used for training the model. By default, the `label_map.json` file will be used.
5. {Optional} If your model is trained on images sizes other than 224, make sure to pass this number as an argument to the script (see [usage](#usage) below).

### Usage

1. Basic usage

```
$ python3 predict.py <PATH/TO/IMAGE/> <PATH/TO/MODEL>

# For example,
$ python3 predict.py test_images/cautleya_spicata.jpg 1725545374.h5
```

2. Predict top K images instead of only 1

```
$ python3 predict.py <PATH/TO/IMAGE/> <PATH/TO/MODEL> --top_k <K>

# For example,
$ python3 predict.py test_images/cautleya_spicata.jpg 1725545374.h5 --top_k 3
```

3. Predict with your own label mapping

```
$ python3 predict.py <PATH/TO/IMAGE/> <PATH/TO/MODEL> --category_names <PATH/TO/LABEL/MAP>

# For example,
$ python3 predict.py test_images/cautleya_spicata.jpg 1725545374.h5 --category_names label_map.json
```