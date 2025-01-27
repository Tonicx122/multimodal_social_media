# -*- coding: utf-8 -*-

"""
用于加载训练好的多模态模型，并对新数据集进行预测。
"""
import csv
import os
import pickle
import warnings

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bin import aidrtokenize
from bin.prediction_data_generator import PredictionDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 预定义路径（请根据实际情况修改）
# MODEL_PATH = "models/info_multimodal_paired_agreed_lab.weights.h5"  # 训练好的模型路径
MODEL_PATH = "models/task_informative_text_img_agreed_lab_train/info_multimodal_paired_agreed_lab.weights.hdf5"  # 训练好的模型路径
TOKENIZER_PATH = "models/task_informative_text_img_agreed_lab_train/info_multimodal_paired_agreed_lab.weights.tokenizer"  # tokenizer 路径
LABEL_ENCODER_PATH = "models/task_informative_text_img_agreed_lab_train/info_multimodal_paired_agreed_lab.weights.label_encoder"  # label_encoder 路径
NEW_DATA_PATH = "data/media_data/tweets_images_2025-01-28_01-15-13.csv"  # 新数据集路径
OUTPUT_PATH = "results/new_data_predictions.txt"  # 预测结果输出路径

# 模型参数（需要与训练时一致）
MAX_SEQUENCE_LENGTH = 25  # 文本序列最大长度
BATCH_SIZE = 32  # 批大小
LABEL_INDEX = 6  # 标签列的索引

def load_model_and_utils(model_path, tokenizer_path, label_encoder_path):
    """
    加载模型、tokenizer 和 label_encoder。
    """
    # 加载模型
    model = load_model(model_path)
    print("模型加载成功。")

    # 加载 tokenizer
    with open(tokenizer_path, "rb") as handle:
        tokenizer = pickle.load(handle)
    print("Tokenizer 加载成功。")

    # 加载 label_encoder
    with open(label_encoder_path, "rb") as handle:
        label_encoder = pickle.load(handle)
    print("Label Encoder 加载成功。")

    return model, tokenizer, label_encoder

def read_dev_data_multimodal(data_file, tokenizer, MAX_SEQUENCE_LENGTH, delim=","):
    """
    读取并预处理新数据文件，适配文件表头为 tweet_id,image_id,tweet_text,image。
    """
    data, image_list, ids = [], [], []

    with open(data_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delim)

        # 跳过表头
        next(reader)

        for row_num, row in enumerate(reader, start=1):
            # 检查列数是否足够
            if len(row) < 4:
                print(f"行 {row_num} 列数不足，跳过：{row}")
                continue

            try:
                # 提取文本内容
                txt = aidrtokenize.tokenize(row[2].strip())  # `tweet_text` 在第 3 列（索引 2）
                if len(txt) < 1:
                    print(f"行 {row_num} 文本无效，跳过：{row[2]}")
                    continue
                data.append(" ".join(txt))

                # 提取图片路径
                image_list.append("data/media_data/" + row[3].strip())  # `image` 在第 4 列（索引 3）

                # 提取 Tweet ID
                ids.append(row[0].strip())  # `tweet_id` 在第 1 列（索引 0）
            except Exception as e:
                print(f"行 {row_num} 处理出错：{e}，跳过")
                continue

    # 将文本数据转换为序列
    sequences = tokenizer.texts_to_sequences(data)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    print('Shape of data tensor:', data.shape)
    return data, image_list, ids

def load_new_data(data_path, tokenizer, max_seq_length):
    """
    加载新数据集。
    """
    # 读取新数据
    # new_x, new_image_list, new_y, new_le, new_labels, ids = data_process.read_dev_data_multimodal(
    #     data_path, tokenizer, max_seq_length, label_index, delim="\t"
    # )
    new_data, new_image_list, ids = read_dev_data_multimodal(
        data_path, tokenizer, max_seq_length
    )
    print("新数据加载成功。")
    return new_data, new_image_list, ids


def preprocess_input_vgg(x):
    """
    VGG16 的预处理函数包装器，用于兼容 3D 张量输入。
    :param x: numpy 3D 数组（单张图像）。
    :return: 预处理后的 numpy 3D 数组。
    """
    X = np.expand_dims(x, axis=0)
    X = preprocess_input(X)
    return X[0]


def generate_image_vec_dict(image_file_list, target_size=(224, 224)):
    """
    动态生成图像特征字典，与您提供的逻辑保持一致。
    :param image_file_list: 图像路径列表。
    :param target_size: 图像目标尺寸（默认 224x224）。
    :return: 包含图像预处理数据的字典。
    """
    image_vec_dict = {}

    for img_path in image_file_list:
        try:
            # 加载图像并调整大小
            img = load_img(img_path, target_size=target_size)
            img_data = img_to_array(img)

            # 预处理图像数据
            img_data = preprocess_input_vgg(img_data)

            # 存储预处理后的图像数据
            image_vec_dict[img_path] = img_data

        except Exception as e:
            warnings.warn(f"Error processing image {img_path}: {e}")
            # 如果图像处理失败，存储零数组以避免中断
            image_vec_dict[img_path] = np.zeros((target_size[0], target_size[1], 3))

    return image_vec_dict



def save_predictions(output_path, predictions, ids, label_encoder):
    """
    保存预测结果。
    """
    # 将预测结果转换为类别标签
    predicted_labels = np.argmax(predictions, axis=1)
    predicted_classes = label_encoder.inverse_transform(predicted_labels)

    # 将结果保存到文件
    with open(output_path, "w") as f:
        for id_, pred_class in zip(ids, predicted_classes):
            f.write(f"{id_}\t{pred_class}\n")
    print(f"预测结果已保存到: {output_path}")

def main():
    # 加载模型和相关工具
    model, tokenizer, label_encoder = load_model_and_utils(MODEL_PATH, TOKENIZER_PATH, LABEL_ENCODER_PATH)

    # 加载新数据
    new_data, new_image_list, ids = load_new_data(
        NEW_DATA_PATH, tokenizer, MAX_SEQUENCE_LENGTH
    )

    # 加载图像数据
    image_vec_dict = generate_image_vec_dict(new_image_list)

    # 创建数据生成器（不需要标签）
    params = {
        "max_seq_length": MAX_SEQUENCE_LENGTH,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
    }
    data_generator = PredictionDataGenerator(new_image_list, new_data, image_vec_dict, **params)

    # 进行预测
    predictions = model.predict(data_generator, verbose=1)
    print("预测完成。")

    # 解码预测结果
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    for ids, label in zip(ids, predicted_labels):
        print(f"ID: {id}, Predicted Label: {label}")

    # 保存预测结果
    save_predictions(OUTPUT_PATH, predictions, ids, label_encoder)



if __name__ == "__main__":
    main()