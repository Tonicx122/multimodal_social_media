import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
import warnings
import datetime
import os, errno


def preprocess_input_vgg(x):
    """
    Wrapper around keras.applications.vgg16.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note: This wrapper ensures compatibility between 3D tensors and
    functions expecting 4D tensors.

    Returns:
    -------
    A numpy 3darray (the preprocessed image).
    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]


class DataGenerator(Sequence):
    """
    Generates data for Keras.
    """

    def __init__(self, image_file_list, text_vec, image_vec_dict, labels, max_seq_length=20, batch_size=32,
                 n_classes=2, shuffle=False):
        """
        Initialization of DataGenerator.
        """
        self.batch_size = batch_size
        self.labels = labels
        self.image_file_list = image_file_list
        self.text_vec = text_vec
        self.image_vec_dict = image_vec_dict
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.max_seq_length = max_seq_length
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_file_list) / float(self.batch_size)))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.image_file_list))

        temp_indexes = self.indexes[start:end]

        images_batch, text_batch, y = self.__data_generation(temp_indexes)

        # return [images_batch, text_batch], y
        return (tf.convert_to_tensor(images_batch, dtype=tf.float32),
                    tf.convert_to_tensor(text_batch, dtype=tf.int32)), \
                   tf.convert_to_tensor(y, dtype=tf.float32)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        """
        self.indexes = np.arange(len(self.image_file_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples.
        """
        y = np.zeros((len(indexes), self.n_classes), dtype=int)
        text_batch = np.zeros((len(indexes), self.max_seq_length), dtype=int)
        images_batch = np.zeros((len(indexes), 224, 224, 3), dtype=float)

        for i, index in enumerate(indexes):
            try:
                image_file_name = str(self.image_file_list[index])

                if image_file_name in self.image_vec_dict:
                    img = self.image_vec_dict[image_file_name]
                    images_batch[i] = img
                    y[i] = self.labels[index]
                    text_batch[i] = self.text_vec[index]
                else:
                    warnings.warn(f"Image {image_file_name} not found in image_vec_dict. Assigning zero array.")
                    images_batch[i] = np.zeros((224, 224, 3))

            except Exception as e:
                warnings.warn(f"Exception in data generation for index {index}: {str(e)}")
                images_batch[i] = np.zeros((224, 224, 3))

        current_images_batch = preprocess_input(images_batch)

        return current_images_batch, text_batch, y
