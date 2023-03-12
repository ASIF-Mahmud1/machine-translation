
**End to End Machine Translation System**

  
  

In this article I will walk you through the machine learning project that is able to translate sentences from hindi to english. This project has all the key components that are necessary for an end to end machine learning pipeline.

This has 3 key components: client , server, machine-learning

  

**client**: a mobile application based on javascript framework : react native. The application communicates with the server via API to translate sentences from hindi to english

  

**server**: build with Fast API , a popular web framework for python. The API endpoint: /predict

is made available that takes sentences as params to make the translation.

  

**machine-learning**: it consist of the program that trains the model; machine translation evaluations (BLEU , ROUGE scores ).

  

**Training the Model**

  

**Training Data**:

We have used the parallel corpus from [IIT Bombay English-Hindi Translation Dataset](https://www.cfilt.iitb.ac.in/iitb_parallel/). The size of our training dataset is 15000. 10% of our data is used for testing

  

**Preprocessing**:

1.  All characters are converted to lowercase
    
2.  Number are removed
    
3.  Then tokenizers are created for both src and target sentences
    
4.  Sentences are encoded to sequences of integers.
    
5.  Each encoded sentence has a fixed length of 15.
    

  

**Model Architecture**:

The model is based on LSTM (Long Short Term Memory) which is a type of recurrent neural network (RNN) . LSTM networks can remember long-term dependencies in sequential data, making them well-suited for machine translation tasks. They are able to do this by using a series of memory cells, which can be selectively updated or forgotten based on the input data. This allows the network to learn patterns in the input data over time, and to use this knowledge to make better translations.

In this project we have used the encoder-decoder architecture for translation. The encoder LSTM network takes in the input sentence and produces a fixed-length vector representation of the sentence, which is then passed to the decoder LSTM network. The decoder LSTM network takes the encoder's output vector as input and generates the translated sentence word by word.