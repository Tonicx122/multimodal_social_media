# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 20:42:24 2017; Feb/2020

@author: Firoj Alam
"""

import numpy as np
import os
import sys
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn import preprocessing
import aidrtokenize  # 假设这是外部依赖模块

# Set seed for reproducibility
np.random.seed(1337)
random.seed(1337)

def file_exist(file_name):
    return os.path.exists(file_name)

def read_stop_words(file_name):
    if not file_exist(file_name):
        print(f"Stop words file not found: {file_name}")
        sys.exit(0)

    stop_words = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                stop_words.append(line)
    return stop_words

stop_words_file = "bin/etc/stop_words_english.txt"
stop_words = read_stop_words(stop_words_file)

def read_train_data(dataFile, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, delim):
    """
    Prepare the training data
    """
    data, labels = [], []
    with open(dataFile, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split(delim)
            txt = aidrtokenize.tokenize(row[3].strip().lower())
            label = row[6]
            if not txt:
                print(f"Empty tokenized text: {txt}")
                continue
            data.append(txt)
            labels.append(label)

    # Shuffle data
    index_shuf = list(range(len(data)))
    random.shuffle(index_shuf)
    data_shuf = [data[i] for i in index_shuf]
    lab_shuf = [labels[i] for i in index_shuf]

    # Encode labels
    le = preprocessing.LabelEncoder()
    yL = le.fit_transform(lab_shuf)
    labels = list(le.classes_)

    yC = len(set(yL))
    yR = len(yL)
    y = np.zeros((yR, yC), dtype=np.int32)
    y[np.arange(yR), yL] = 1

    # Tokenize and pad sequences
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, oov_token="OOV_TOK")
    tokenizer.fit_on_texts(data_shuf)
    sequences = tokenizer.texts_to_sequences(data_shuf)
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens.")

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(f"Shape of data tensor: {data.shape}")
    return data, y, le, labels, word_index, tokenizer

def read_dev_data(dataFile, tokenizer, MAX_SEQUENCE_LENGTH, delim, train_le):
    """
    Prepare the development/validation data
    """
    data, labels, id_list = [], [], []
    with open(dataFile, 'r', encoding='utf-8') as f:
        next(f)  # Skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = line.split(delim)
            t_id = row[2].strip().lower()
            txt = aidrtokenize.tokenize(row[3].strip().lower())
            label = row[6]
            if not txt:
                print(f"Empty tokenized text: {txt}")
                continue
            data.append(txt)
            labels.append(label)
            id_list.append(t_id)

    print(f"Number of validation samples: {len(data)}")

    # Encode labels
    yL = train_le.transform(labels)
    yC = len(set(yL))
    yR = len(yL)
    y = np.zeros((yR, yC), dtype=np.int32)
    y[np.arange(yR), yL] = 1

    # Tokenize and pad sequences
    sequences = tokenizer.texts_to_sequences(data)
    word_index = tokenizer.word_index
    print(f"Found {len(word_index)} unique tokens.")
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    print(f"Shape of data tensor: {data.shape}")
    return data, y, train_le, labels, word_index, id_list

def load_embedding(fileName):
    print('Indexing word vectors.')
    embeddings_index = {}
    with open(fileName, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f"Found {len(embeddings_index)} word vectors.")
    return embeddings_index

def prepare_embedding(word_index, model, MAX_NB_WORDS, EMBEDDING_DIM):
    """
    Prepare embedding matrix
    """
    nb_words = min(MAX_NB_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM), dtype=np.float32)

    for word, i in word_index.items():
        if i >= nb_words:
            continue
        if word in model:
            embedding_vector = model[word][:EMBEDDING_DIM]
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    return embedding_matrix

def str_to_indexes(s):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    input_size = 1014
    char_dict = {char: idx + 1 for idx, char in enumerate(alphabet)}

    s = s.lower()
    max_length = min(len(s), input_size)
    str2idx = np.zeros(input_size, dtype='int64')
    for i in range(1, max_length + 1):
        c = s[-i]
        if c in char_dict:
            str2idx[i - 1] = char_dict[c]
    return str2idx
