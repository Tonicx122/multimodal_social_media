import pickle
import warnings
import datetime
import argparse  # 使用 argparse 替代 optparse
import os
import errno
import tensorflow as tf  # 使用 TensorFlow 的 Keras 库
from tensorflow.keras.preprocessing import image  # 更新为 tensorflow.keras.preprocessing
from tensorflow.keras.applications.vgg16 import preprocess_input  # 更新为 tensorflow.keras.applications
import numpy as np


def preprocess_input_vgg(x):
    """Wrapper around tensorflow.keras.applications.vgg16.preprocess_input()
    to make it compatible for use with tensorflow.keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass tensorflow.keras.applications.vgg16.preprocess_input()
    directly to tensorflow.keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).
    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]

def read_train_data_multimodal(data_file, delim="\t"):
    """
    Prepare the data
    """
    image_list = []
    with open(data_file, 'r') as f:  # 'rU' 在 Python 3 中已废弃，直接使用 'r'
        for line in f:
            line = line.strip()
            if line == "":
                continue
            row = line.split(delim)
            image_path = row[0].strip()
            image_list.append(image_path)

    return image_list

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # 使用 argparse 替代 optparse
    parser = argparse.ArgumentParser(description="Process some images.")
    parser.add_argument('-i', '--input_file_name', type=str, required=True, help="Input file name")
    parser.add_argument('-o', '--output_file_name', type=str, required=True, help="Output file name")

    options = parser.parse_args()

    a = datetime.datetime.now().replace(microsecond=0)

    input_file_name = options.input_file_name
    output_file_name = options.output_file_name
    data = read_train_data_multimodal(input_file_name)

    images_npy_data = {}  # np.empty([len(instances), 224, 224, 3])
    for i, img_path in enumerate(data):
        img = image.load_img(img_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        images_npy_data[img_path] = img_data

    # 将数据保存为 pickle 文件
    with open(output_file_name, 'wb') as handle:
        pickle.dump(images_npy_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
