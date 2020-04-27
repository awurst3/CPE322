import numpy as np
import re
import os

def get_hyperparameters(param_path):
    '''Load the relevant parameters used in training the model for testing
    The file specified by param_path consists of five lines containing:
    - total_words
    - embedding_dim
    - max_length (int)
    - dropout_factor
    - num_epochs
    '''
    
    with open(param_path, 'r') as f:
        params = [float(f.readline()) for i in range(5)]
    
    return int(params[0]), int(params[1]), int(params[2]), params[3], int(params[4])

def get_test_sequences(model_dir):
    '''Load the test sequences (data and labels) split from the full dataset during training'''
    
    x_test = np.load(model_dir + 'x_test.npy')
    y_test = np.load(model_dir + 'y_test.npy')
    return x_test, y_test

def get_weights_file(model_dir, num_epochs):
    '''Get the name of the file that contains the model weights (based on the last training epoch)'''
    
    pattern = 'ep'+str(num_epochs)+'[\d\w._-]*.hdf5'
    matches = [re.search(pattern, file) for file in os.listdir(model_dir)]
    return [i.group(0) for i in matches if i is not None][0]
