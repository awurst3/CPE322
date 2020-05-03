# Fake Review Detection Using Deep Learning

Deep learning methods are very powerful in the context of natural language processing (NLP), which includes the problem of finding fake reviews. The data is text-based, and so a deep learning model can be constructed to accurately classify deceptive and truthful reviews. First, however, the data must be preprocessed such that it can be used effectively in a neural network. The entire process is explained below. An accuracy of 90.0% was achieved using this deep neural network. 

## Data and Preprocessing

The [Deceptive Opinion Spam Corpus](https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus)<sup>1</sup> was used to train and test the neural network model. The dataset contains 1600 reviews for hotels, where 800 are truthful and 800 are deceptive. Because there are two classes of reviews, truthful labels were assigned a value of zero while deceptive reviews were assigned a value of one. For the text data itself, a tokenizer was used to convert the sequences of words to sequences of numbers, or tokens. Each word is given a unique token and replaced by its token. Because each review can have a different length, the sequences were also padded on the right with zeros to match the longest sequence's size. With uniformly size data sequences and the binary sequence of labels, the data could be inputted to a neural network. 

## Neural Network Architecture

Because the input data is text-based, an RNN makes sense to use in order to classify the reviews. The first hidden layer in the model is a word embedding layer. This layer's role is to represent each word with an *n*-dimensional vector, where *n* can be tuned based on the number of features that need to be captured. In this project, *n* was selected to be 16 due to the length of the reviews, but further tuning may show a better value instead. Following the embedding layer is a dropout layer, which is used to regularize the network and prevent overfitting. The dropout factor was set to 0.5 and can be experimented with just like the embedding dimension.

Next is a series of convolutional, max pooling, and dropout layers. A convolutional layer is useful in reducing the number of parameters in the input data while still preserving its features. Max pooling is a technique further decreases the size of the data by taking the maximum of every pair of values, cutting the size of the data in half. A dropout layer is used again for regularization afterwards. This series of a convolutional, max pooling, and dropout layers is repeated three times to continue changing the data's size before entering the last stage of the network.

Finally, a long short-term memory (LSTM) layer, a type of recurrent neural network layer, is used. More specifically, a bidirectional LSTM (BLSTM) is used with a dropout factor of 0.5. The BLSTM is effectively two unidirectional LSTMs stacked together, where one propagates forwards while the other propagates backwards. This is useful because each LSTM is capable of capturing long term context in the text data. By using a BLSTM, this capability is made even better. This is the final hidden layer before the output layer, which contains a single node with a *sigmoid* activation function. This function classifies the input as 0 (truthful) or 1 (deceptive). 

## Results

After splitting the 1600 data examples into 85% training/validation and 15% testing, the model's accuracy was 90.0%. While training the model using various hyperparameter settings, such as different embedding dimension or dropout factor values, the model converged after only five epochs usually. Additionally, the loss started to increase after converging, while the accuracy fluctuated slightly in the epochs to follow, possibly indicating overfitting. More modifications and tuning can be done to further improve the performance. 

## References

[1] M. Ott, C. Cardie, and J.T. Hancock, "Deceptive Opinion Spam Corpus." May 2017. Distributed by Kaggle. https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus

___
I pledge my honor that I have abided by the Stevens Honor System
