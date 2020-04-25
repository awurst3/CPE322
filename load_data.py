import numpy as np
from random import shuffle
from pandas import read_csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def split_data(data, labels, test_fraction):
    '''Split input data into training and testing sets'''

    # Load integers from 0 to the number of data samples into an array
    # These integers will be shuffled and used as the indices of the resulting split arrays
    indices = np.arange(len(data))
    shuffle(indices)
    indices = np.array(indices)

    # Fill new arrays with their elements randomly shuffled
    divider = round(test_fraction * len(data))
    test_data = [data[indices[i]] for i in range(divider)]
    test_labels = [labels[indices[i]] for i in range(divider)]
    train_data = [data[indices[i]] for i in range(divider,len(data))]
    train_labels = [labels[indices[i]] for i in range(divider,len(data))]

    return train_data, train_labels, test_data, test_labels

    

def load_data(filename, data_column, label_column, label_names, test_split, val_split):
    '''Extract the data from the dataset and convert the text to sequences'''

    # Read the file in as a dataframe and extract the data and labels from it
    df = read_csv(filename).dropna()
    data = df[data_column]
    labels = df[label_column]
    max_length = max([len(i) for i in data])

    train_num = len(data) * (1 - test_split) * (1 - val_split)
    val_num = len(data) * (1 - test_split) * val_split
    
    
    # Assign a unique token (integer) to each word found in the data
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(data)
    word_index = tokenizer.word_index
    total_words = len(word_index)

    # Convert each sequence of words into a tokenized equence of integers
    # Pad the sequences to the maximum input length so the data is uniformly sized
    data_seq = tokenizer.texts_to_sequences(data)
    padded_data_seq = np.array(pad_sequences(data_seq,
                                             padding='post',
                                             maxlen=max_length))

    # Since there are only two labels, the label sequence can easily be constructed using list comprehension
    label_seq = [[0] if label==label_names[0] else [1] if label==label_names[1] else [-1] for label in labels]
    label_word_index = {label_names[0]: 0, label_names[1]: 1}
    
    # Need to split data into training and testing sets
    train_data, train_labels, test_data, test_labels = split_data(padded_data_seq,
                                                                  label_seq,
                                                                  test_split)

    # Convert the sequences into numpy arrays so that they can be used in the neural network
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels, \
           max_length, total_words, train_num, val_num
