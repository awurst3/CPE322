# Fake Review Detection Using Deep Learning

Reviews offer a great way to help guide one's search for a specific product. However, many online product pages contain fake reviews, which can make it harder to for consumers to decide whether or not to make a purchase. This project attempts to address this issue by detecting fake reviews using artificial intelligence. By using a labelled review dataset and a multilayer recurrent neural network (RNN), fake reviews are detected with 87.5% accuracy. The ultimate goal of the project is to build a browser tool to identify fake reviews using a neural network trained on fake review data. 

## Data and Preprocessing

The [Deceptive Opinion Spam Corpus](https://www.kaggle.com/rtatman/deceptive-opinion-spam-corpus) was used to train and test the neural network model. The dataset contains 1600 reviews for hotels, where 800 are truthful and 800 are deceptive. Because there are two classes of reviews, truthful labels were assigned a value of zero while deceptive reviews were assigned a value of one. For the text data itself, a tokenizer was used to convert the sequences of words to sequences of numbers, or tokens. Each word is given a unique token and replaced by its token. Because each review can have a different length, the sequences were also padded on the right with zeros to match the longest sequence's size. With uniformly size data sequences and the binary sequence of labels, the data could be inputted to a neural network. 

## Neural Network Architecture

Because the input data is text-based, an RNN makes sense to use in order to classify the reviews. The first hidden layer in the model is a word embedding layer. This layer's role is to represent each word with an *n*-dimensional vector, where *n* can be tuned based on the number of features that need to be captured. In this project, *n* was selected to be 16 due to the length of the reviews, but further tuning may show a better value instead. Following the embedding layer is a dropout layer, which is used to regularize the network and prevent ovefitting. The dropout factor was set to 0.5, and can be experimented with just like the embedding dimension.

Next, a series of long short-term memory (LSTM) layers is used. More specifically, two bidirectional LSTM (BLSTM) layers are used together, each with another dropout layer following their outputs. Each BLSTM layer is broken up into a forward-propogating layer of LSTM cells and a backwards-propogating layer of LSTM cells. Each pair of corresponding forward and backward LSTM cells produce a single output that incorporates context in the text. Again, the dropout layers are placed to prevent overfitting. 

## Results

After splitting the 1600 data examples into 85% training/validation and 15% testing, the model's accuracy was 87.5%. 


