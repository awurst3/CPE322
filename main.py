import load_data
import metrics
from model import Model

from os import mkdir, environ
from time import time, strftime, localtime
from numpy import save

def main():
    # Hyper-parameters (to be tuned)
    num_epochs = 30
    embedding_dim = 16
    val_split = 0.15
    test_split = 0.15
    batch_sz = 32
    dropout_factor = 0.5
    params = ['accuracy', 'loss']

    # Set up the output directory for the weights and accuracies to go
    output_dir = 'Training_' + strftime('%Y_%m_%d_%Hh%Mm%Ss', localtime(time()))
    mkdir(output_dir)
    
    environ["CUDA_VISIBLE_DEVICES"] = "1"  # Change depending on hardware

    # Load the data from the .csv file
    train_data, train_labels, test_data, test_labels, max_length, total_words, train_num, val_num = \
        load_data.load_data('C:\\CPE322\\deceptive-opinion.csv',
                            'text',
                            'deceptive',
                            ['truthful', 'deceptive'],
                            test_split,
                            val_split)

    # Save the model parameters so they can be used when loading the model
    with open(output_dir+'\model_params.txt', 'w') as f:
        f.write(str(total_words) + '\n' + \
                str(embedding_dim) + '\n' + \
                str(max_length) + '\n' + \
                str(dropout_factor) + '\n' + \
                str(num_epochs))

    # Initialize model parameters
    model = Model(train_data,
                  train_labels,
                  test_data,
                  test_labels,
                  val_split,
                  total_words+1,  # Add one for out-of-vocabulary
                  embedding_dim,
                  max_length,
                  dropout_factor,
                  num_epochs,
                  batch_sz,
                  output_dir)

    model.build()
    model.set_checkpoint_callback('val_accuracy')  # Change to 'val_acc' depending on Keras version
    model.set_log_callback()
    
    # Save the test data split in this run so that the model tests on different data than it was trained on
    model.save_test_sequences()

    # Print model and data information
    model.summary()
    model.save_summary()
    
    print('Training Sample Num: %d\n'\
          'Validation Sample Num: %d\n'\
          'Feature Size: %d\n'\
          'Saving results to: %s\n'
          % (train_num, val_num, max_length, output_dir))

    start = time()

    # Compile and train the model, saving the history logs so they can be reused later
    model.compile()
    history = model.train()

    end = time()
    print('Total time: %s s' % (end - start))

    # Show plots of accuracy and loss over the course of the training epochs
    # These plots will be saved automatically as well
    metrics.plot(history, params, output_dir)

    

if __name__ == '__main__':
    main()
