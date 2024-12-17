# -*- coding: utf-8 -*-

#

"""
Created on Sun Apr  2 10:33:22 2017; Feb/2020

@author: Firoj Alam, Ferda Ofli
Adopted from:
# https://nbviewer.jupyter.org/gist/embanner/6149bba89c174af3bfd69537b72bca74
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

"""
import optparse

import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input, BatchNormalization, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger, TensorBoard
import warnings
import datetime
import os
import pickle
import numpy as np
from time import time
import data_process_multimodal_pair as data_process
import performance
from crisis_data_generator_image_optimized import DataGenerator
import cnn_filter as cnn_filter
from gensim.models import KeyedVectors

class ImgInstance(object):
    def __init__(self, id=1, imgpath="", label=""):
        self.id = id
        self.imgpath = imgpath
        self.label = label

def resnet_model():
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    last_layer_output = Flatten(name="resnet_flatten")(resnet.output)
    return last_layer_output, resnet

def vgg_model():
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # last_layer_output = vgg16.get_layer('fc2').output
    last_layer_output = Flatten(name="vgg16_flatten")(vgg16.output) # 添加唯一名称
    return last_layer_output, vgg16

def check_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def file_exist(w2v_checkpoint):
    return os.path.exists(w2v_checkpoint)

def save_model(model, model_dir, model_file_name, tokenizer, label_encoder):
    os.makedirs(model_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(model_file_name))[0]
    model_file = os.path.join(model_dir, base_name + ".hdf5")
    tokenizer_file = os.path.join(model_dir, base_name + ".tokenizer")
    label_encoder_file = os.path.join(model_dir, base_name + ".label_encoder")

    with open(tokenizer_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(label_encoder_file, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(model_file)

def write_results(out_file, file_name, accu, P, R, F1, wAUC, AUC, report, conf_mat):
    result = (
        f"{AUC * 100:.2f}\t{accu * 100:.2f}\t{P * 100:.2f}\t{R * 100:.2f}\t{F1 * 100:.2f}\n"
    )
    print(result)
    print(report)
    out_file.write(file_name + "\n")
    out_file.write(result)
    out_file.write(report)
    out_file.write(conf_mat)

def dir_exist(dirname):
    os.makedirs(dirname, exist_ok=True)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = optparse.OptionParser()
    parser.add_option('-i', action="store", dest="train_data", default=None, type="string")
    parser.add_option('-v', action="store", dest="val_data", default=None, type="string")
    parser.add_option('-t', action="store", dest="test_data", default=None, type="string")
    parser.add_option('-m', action="store", dest="model_file", default="best_model.hdf5", type="string")
    parser.add_option('-o', action="store", dest="outputfile", default="results.tsv", type="string")
    parser.add_option("-w", "--w2v_checkpoint", action="store", dest="w2v_checkpoint",
                      default="w2v_models/data_w2v_info.model", type="string")
    parser.add_option("-d", "--log_dir", action="store", dest="log_dir", default="model_log/", type="string")
    # parser.add_option("-l","--log_file", action="store", dest="log_file", default="./log", type="string")
    parser.add_option("-c", "--checkpoint_log", action="store", dest="checkpoint_log", default="./checkpoint_log/",
                      type="string")
    parser.add_option("-x", "--vocab_size", action="store", dest="vocab_size", default=20000, type="int")
    parser.add_option("--embedding_dim", action="store", dest="embedding_dim", default=300, type="int")
    parser.add_option("--batch_size", action="store", dest="batch_size", default=32, type="int")
    parser.add_option("--nb_epoch", action="store", dest="nb_epoch", default=1000, type="int")
    parser.add_option("--max_seq_length", action="store", dest="max_seq_length", default=25, type="int")
    parser.add_option("--patience", action="store", dest="patience", default=100, type="int")
    parser.add_option("--patience-lr", action="store", dest="patience_lr", default=10, type="int")
    # parser.add_option("-n", "--num_of_inst", action="store", dest="num_of_inst", default=10, type="int")
    parser.add_option("--text_sim_score", action="store", dest="text_sim_score", default=0.6, type="float")
    parser.add_option("--image_sim_score", action="store", dest="image_sim_score", default=0.6, type="float")
    parser.add_option("--total_sim_score", action="store", dest="total_sim_score", default=0.6, type="float")
    parser.add_option("--label_index", action="store", dest="label_index", default=6, type="int")
    parser.add_option("--image_dump", action="store", dest="image_dump",
                      default="data/task_data/all_images_data_dump.npy", type="string")

    options, args = parser.parse_args()
    a = datetime.datetime.now().replace(microsecond=0)

    train_file = options.train_data
    val_file = options.val_data
    test_file = options.test_data
    out_file = options.outputfile
    best_model_path = options.model_file
    log_path = options.checkpoint_log
    log_dir = os.path.abspath(os.path.dirname(log_path))
    dir_exist(log_dir)

    base_name = os.path.basename(train_file)
    base_name = os.path.splitext(base_name)[0]
    log_file = log_dir + "/" + base_name + "_log_v2.txt"

    with open(options.image_dump, 'rb') as handle:
        images_npy_data = pickle.load(handle)

    ######## Parameters ########
    # MAX_SEQUENCE_LENGTH = 25
    # MAX_NB_WORDS = 20000
    # EMBEDDING_DIM = 300
    # batch_size = 32
    # nb_epoch = 1000
    # patience_early_stop = 100
    # patience_learning_rate = 10
    MAX_SEQUENCE_LENGTH = options.max_seq_length
    MAX_NB_WORDS = options.vocab_size
    EMBEDDING_DIM = options.embedding_dim
    batch_size = options.batch_size
    nb_epoch = options.nb_epoch
    patience_early_stop = options.patience
    patience_learning_rate = options.patience
    dir_exist(options.checkpoint_log)
    delim = "\t"

    ######## Dataset Preparation ########

    #### training dataset
    dir_name = os.path.dirname(train_file)
    base_name = os.path.splitext(os.path.basename(train_file))[0]

    train_x, train_image_list, train_y, train_le, train_labels, word_index, tokenizer = data_process.read_train_data_multimodal(
        train_file, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, int(options.label_index), delim
    )

    #### development dataset
    base_name = os.path.splitext(os.path.basename(val_file))[0]

    dev_x, dev_image_list, dev_y, dev_le, dev_labels, _ = data_process.read_dev_data_multimodal(
        val_file, tokenizer, MAX_SEQUENCE_LENGTH, int(options.label_index), delim
    )

    nb_classes = len(set(train_labels))
    print(f"Number of classes: {nb_classes}")

    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size, "n_classes": nb_classes, "shuffle": True}
    train_data_generator = DataGenerator(train_image_list, train_x, None, train_y, **params)

    params = {"max_seq_length": MAX_SEQUENCE_LENGTH, "batch_size": batch_size, "n_classes": nb_classes, "shuffle": False}
    val_data_generator = DataGenerator(dev_image_list, dev_x, None, dev_y, **params)

    MAX_SEQUENCE_LENGTH = options.max_seq_length
    MAX_NB_WORDS = options.vocab_size

    ######## Word-Embedding ########
    word_vec_model_file = "crisisNLP_word_vector.bin"
    emb_model = KeyedVectors.load_word2vec_format(word_vec_model_file, binary=True)
    embedding_matrix = data_process.prepare_embedding(word_index, emb_model, MAX_NB_WORDS, EMBEDDING_DIM)
    # options.emb_matrix = embedding_matrix
    # options.vocab_size, options.embedding_dim = embedding_matrix.shape
    # pickle.dump(options.emb_matrix, open(options.w2v_checkpoint, "wb"))


    ######## Text Network ########
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,))
    cnn = cnn_filter.kimCNN(embedding_matrix, word_index, MAX_NB_WORDS, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, text_input)
    text_network = Dense(1000, activation="relu", name="text_dense_1")(cnn)
    text_network = BatchNormalization(name="text_bn")(text_network)

    ######## Image Network ########
    image_output, vgg16 = vgg_model()
    image_network = Dense(1000, activation="relu", name="img_dense_1")(image_output)
    image_network = BatchNormalization(name="img_bn")(image_network)

    ######## Combined Network ########
    merged_network = concatenate([image_network, text_network], axis=-1)
    merged_network = Dropout(0.4, name="dropout_1")(merged_network)
    merged_network = Dense(500, activation="relu", name="dense_1")(merged_network)
    merged_network = Dropout(0.2, name="dropout_2")(merged_network)
    merged_network = Dense(100, activation="relu", name="dense_2")(merged_network)
    merged_network = Dropout(0.02, name="dropout_3")(merged_network)
    output = Dense(nb_classes, activation="softmax", name="output")(merged_network)

    model = Model(inputs=[vgg16.input, text_input], outputs=output)

    ######## Model Compilation ########
    lr = 1e-5
    print("lr= " + str(lr) + ", beta_1=0.9, beta_2=0.999, amsgrad=False")
    adam = Adam(learning_rate=lr,beta_1=0.9,beta_2=0.999,amsgrad=False)
    model.compile(optimizer=adam, loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())

    ######## Callbacks ########
    callbacks_list = [
        EarlyStopping(monitor="val_accuracy", patience=patience_early_stop, verbose=1, mode="max"),
        ReduceLROnPlateau(monitor="val_accuracy", patience=patience_learning_rate, verbose=1, factor=0.1, min_lr=1e-6),
        CSVLogger("training_log.csv", append=False, separator="\t"),
        ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, mode="max"),
        TensorBoard(log_dir=f"logs/{time()}", histogram_freq=1),
    ]

    ######## Model Training ########
    history = model.fit(
        train_data_generator, epochs=nb_epoch, validation_data=val_data_generator, verbose=1, callbacks=callbacks_list
    )

    ######## Save the model ########
    print("Saving model...")
    model.load_weights(best_model_path)
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    print("Best saved model loaded...")

    # 保存模型和相关文件
    dir_name = os.path.dirname(best_model_path)
    base_name = os.path.splitext(os.path.basename(train_file))[0]
    model_dir = os.path.join(dir_name, base_name)
    save_model(model, model_dir, best_model_path, tokenizer, train_le)

    ############ Test data ########
    # 生成输出文件路径
    dir_name = os.path.dirname(out_file)
    base_name = os.path.splitext(os.path.basename(out_file))[0]
    out_file_name = os.path.join(dir_name, base_name + ".txt")

    # 打开输出文件
    with open(out_file_name, "w") as out_file:
        # 开始预测
        dev_prob = model.predict(val_data_generator, verbose=1)
        print(f"dev true len: {len(dev_y)}")
        print(f"dev pred len: {len(dev_prob)}")

        # 计算性能指标
        AUC, accu, P, R, F1, report = performance.performance_measure_cnn(dev_y, dev_prob, train_le)
        result = f"{accu:.4f}\t{P:.4f}\t{R:.4f}\t{F1:.4f}\t{AUC:.4f}\n"
        print(result)
        print(report)

        # 写入结果
        out_file.write(f"{val_file}\n")
        out_file.write(result)
        out_file.write(report)

        # 加载测试数据
        test_x, test_image_list, test_y, test_le, test_labels, ids = data_process.read_dev_data_multimodal(
            test_file, tokenizer, MAX_SEQUENCE_LENGTH, int(options.label_index), delim
        )

        print(f"Number of classes: {nb_classes}")
        params = {
            "max_seq_length": MAX_SEQUENCE_LENGTH,
            "batch_size": batch_size,
            "n_classes": nb_classes,
            "shuffle": False,
        }
        print(f"image size: {len(test_image_list)}")
        print(f"test x: {len(test_x)}")
        print(f"test y: {len(test_y)}")

        # 创建测试数据生成器
        test_data_generator = DataGenerator(test_image_list, test_x, images_npy_data, test_y, **params)

        ######## Evaluation ########
        test_prob = model.predict(test_data_generator, verbose=1)
        print(f"test true len: {len(test_y)}")
        print(f"test pred len: {len(test_prob)}")

        # 计算性能指标
        AUC, accu, P, R, F1, report = performance.performance_measure_cnn(dev_y, dev_prob, train_le)
        result = f"{accu:.4f}\t{P:.4f}\t{R:.4f}\t{F1:.4f}\t{AUC:.4f}\n"
        print(result)
        print(report)

        # 写入结果
        out_file.write(f"{val_file}\n")
        out_file.write(result)
        out_file.write(report)

        # 再次评估测试集
        AUC, accu, P, R, F1, report = performance.performance_measure_cnn(test_y, test_prob, train_le)
        result = f"{accu:.4f}\t{P:.4f}\t{R:.2f}\t{F1:.4f}\t{AUC:.4f}\n"
        print(f"results-cnn:\t{base_name}\t{result}")
        print(report)
        out_file.write(f"{test_file}\n")
        out_file.write(result)
        out_file.write(report)

        # 写入混淆矩阵
        conf_mat_str = performance.format_conf_mat(test_y, test_prob, train_le)
        out_file.write(conf_mat_str + "\n")

    ######## Time measurement ########
    b = datetime.datetime.now().replace(microsecond=0)
    print("time taken:")
    print(b - a)
