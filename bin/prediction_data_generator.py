import warnings

import numpy as np
# import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class PredictionDataGenerator(Sequence):
    """
    Data generator for prediction, generating batches of image and text inputs without labels.
    """

    def __init__(self, image_file_list, text_vec, image_vec_dict, batch_size=32, max_seq_length=25, shuffle=False):
        """
        Initialize the prediction data generator.
        :param image_file_list: List of image file paths.
        :param text_vec: Numpy array of text sequences.
        :param image_vec_dict: Dictionary mapping image file names to precomputed image feature vectors.
        :param batch_size: Number of samples per batch.
        :param max_seq_length: Maximum sequence length for text data.
        :param shuffle: Whether to shuffle data between epochs.
        """
        self.image_file_list = image_file_list
        self.text_vec = text_vec
        self.image_vec_dict = image_vec_dict
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the number of batches per epoch.
        """
        return int(np.ceil(len(self.image_file_list) / float(self.batch_size)))

    def __getitem__(self, index):
        """
        Generate one batch of data.
        """
        start = index * self.batch_size
        end = min((index + 1) * self.batch_size, len(self.image_file_list))

        temp_indexes = self.indexes[start:end]
        images_batch, text_batch = self.__data_generation(temp_indexes)

        return (tf.convert_to_tensor(images_batch, dtype=tf.float32),
                tf.convert_to_tensor(text_batch, dtype=tf.int32))

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
        text_batch = np.zeros((len(indexes), self.max_seq_length), dtype=int)
        images_batch = np.zeros((len(indexes), 224, 224, 3), dtype=float)

        for i, index in enumerate(indexes):
            try:
                image_file_name = str(self.image_file_list[index])

                # Load image features from precomputed dictionary
                if image_file_name in self.image_vec_dict:
                    img = self.image_vec_dict[image_file_name]
                    if img is None:
                        warnings.warn(f"Image vector for {image_file_name} is None. Assigning zero array.")
                        images_batch[i] = np.zeros((224, 224, 3))
                    else:
                        images_batch[i] = img
                else:
                    warnings.warn(f"Image {image_file_name} not found in image_vec_dict. Assigning zero array.")
                    images_batch[i] = np.zeros((224, 224, 3))

                # Load text data
                text_batch[i] = self.text_vec[index]

            except Exception as e:
                warnings.warn(f"Exception in data generation for index {index}: {str(e)}")
                images_batch[i] = np.zeros((224, 224, 3))

        # Preprocess images using VGG's preprocessing function
        current_images_batch = preprocess_input(images_batch)

        return current_images_batch, text_batch
