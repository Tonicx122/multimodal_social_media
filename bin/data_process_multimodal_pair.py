# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017; Feb/2020

@author: Firoj Alam
"""


import numpy as np
import os
import re
import sys
import random
from collections import Counter
from sklearn import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import aidrtokenize as aidrtokenize

# 设置随机种子
np.random.seed(1337)
random.seed(1337)

def file_exist(file_name):
    return os.path.exists(file_name)

def read_stop_words(file_name):
    if not file_exist(file_name):
        print(f"Please check the file for stop words, it is not in provided location {file_name}")
        sys.exit(0)
    with open(file_name, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

stop_words_file = "bin/etc/stop_words_english.txt"
stop_words = read_stop_words(stop_words_file)

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_tokenizer(data, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH):
    """
    Prepare the tokenizer and sequence data.
    """
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print('Shape of data tensor:', data.shape)
    return word_index, tokenizer

def read_train_data_multimodal(data_file, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, label_index, delim):
    """
    Read and preprocess training data.
    """
    data, image_list, lab = [], [], []
    with open(data_file, 'rb') as f:
        next(f)
        for line in f:
            line = line.decode(encoding='utf-8').strip()
            if not line:
                continue
            row = line.split(delim)
            txt = aidrtokenize.tokenize(row[3].strip())
            if len(txt) < 1:
                print("TEXT SIZE:", txt)
                continue
            data.append(" ".join(txt))
            image_list.append(row[4].strip())
            lab.append(row[int(label_index)].strip())

    counts = Counter(lab)
    print(counts)

    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    data_shuf = [data[i] for i in index_shuf]
    image_list_shuf = [image_list[i] for i in index_shuf]
    lab_shuf = [lab[i] for i in index_shuf]

    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab_shuf)
    labels = list(le.classes_)
    print("Training classes:", " ".join(labels))

    y = np.zeros((len(yL), len(labels)), dtype=np.int32)
    y[np.arange(len(yL)), yL] = 1

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
    tokenizer.fit_on_texts(data_shuf)
    sequences = tokenizer.texts_to_sequences(data_shuf)

    word_index = tokenizer.word_index
    print(f'Found {len(word_index)} unique tokens.')
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print('Shape of data tensor:', data.shape)

    return data, image_list_shuf, y, le, labels, word_index, tokenizer

def read_dev_data_multimodal(data_file, tokenizer, MAX_SEQUENCE_LENGTH, label_index, delim):
    """
    Read and preprocess development data.
    """
    data, image_list, lab, ids = [], [], [], []
    with open(data_file, 'rb') as f:
        next(f)
        for line in f:
            line = line.decode(encoding='utf-8').strip()
            if not line:
                continue
            row = line.split(delim)
            txt = aidrtokenize.tokenize(row[3].strip())
            if len(txt) < 1:
                print("TEXT SIZE:", txt)
                continue
            data.append(" ".join(txt))
            lab.append(row[int(label_index)].strip())
            image_list.append(row[4].strip())
            ids.append(row[2].strip())

    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab)
    labels = list(le.classes_)
    print("Training classes:", " ".join(labels))

    y = np.zeros((len(yL), len(labels)), dtype=np.int32)
    y[np.arange(len(yL)), yL] = 1

    sequences = tokenizer.texts_to_sequences(data)
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    print('Shape of data tensor:', data.shape)

    return data, image_list, y, le, labels, ids

def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
    """
    Prepare the embedding matrix.
    """
    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype=np.float32)
    rng = np.random.default_rng()
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        # embedding_vector = model.get(word, rng.normal(size=EMBEDDING_DIM))
        if word in model:
            embedding_vector = model[word]
        else:
            embedding_vector = rng.normal(size=EMBEDDING_DIM)
        embedding_matrix[i] = embedding_vector
    return embedding_matrix
