"""
This file contains a Python script implementing a sequence-to-sequence translation model using LSTM (Long Short-Term Memory) neural networks with the Keras library. 
The purpose of this script is to train a model for translating text from one language to another.

The Translator class defined in this script provides methods for training the translation model, saving and loading the trained model and associated parameters, generating predictions on test data, and comparing the model's predictions with the actual target sequences.

The main functionalities of this script include:

‣  Cleaning input text data.
‣  Preprocessing the text data to prepare it for training.
‣  Creating and compiling the sequence-to-sequence model architecture.
‣  Training the model on the prepared data.
‣  Saving the trained model and its parameters for future use.
‣  Loading a saved model and associated parameters.
‣  Generating predictions on a test dataset and comparing them with the ground truth.

To use this script, users need to set the desired training size before running the script. After execution, the script will train the translation model and save it along with its parameters. Additionally, users can load the trained model, generate predictions on new data, and evaluate the model's performance.

"""

import os
import sys
import re
from string import punctuation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential,load_model
from keras.layers import LSTM,Dense,Embedding,RepeatVector,TimeDistributed
from keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
sys.path.append('../data')
from iit_dataset import createDataset
print("Train LSTM", tf.__version__)

TRAINING_SIZE=5000

class Translator():

    def __init__(self, training_size=10000) -> None:

        """Initialize the Translator object.

        Args:
            training_size (int, optional): Number of training samples to use. Defaults to 10000.
        """
        self.model=None
        self.training_size=training_size
        self.idx_src=0
        self.idx_tar = 1
        self.source_str, self.target_str = "Hindi", "English"
        self.tar_tokenizer=None #
        self.tar_vocab_size=None #
        self.src_tokenizer=None #
        self.tar_length=None #
        self.src_vocab_size=None #
        self.src_length=None #
        self.trainX=None
        self.trainY=None
        pass

    def _get_training_data(self):

        """Fetch training data."""
        pool_oftexts, pairs =createDataset(data_size=self.training_size, type="train")
        dataset= pool_oftexts
        return dataset
    
    def clean(self,string):

        """Clean the input string.

        Args:
            string (str): Input string to be cleaned.

        Returns:
            str: Cleaned string.
        """

        string = string.replace("\u202f"," ")
        string = string.lower()
        for p in punctuation + "«»" + "0123456789":
            string = string.replace(p," ")  
        string = re.sub('\s+',' ', string)
        string = string.strip()
            
        return string
    
    def _generate_train_test_split(self):

        """Generate training and testing dataset splits.

        Returns:
            tuple: Tuple containing dataset, training set, and testing set.
        """
        
        dataset= self._get_training_data()
        total_sentences= len(dataset)
        test_proportion = 0.1
        train_test_threshold = int( (1-test_proportion) * total_sentences)

        dataset["eng"] = dataset["eng"].apply(lambda x: self.clean(x))
        dataset["hindi"] = dataset["hindi"].apply(lambda x: self.clean(x))

        dataset = dataset.values
        dataset = dataset[:total_sentences]

        train, test = dataset[:train_test_threshold], dataset[train_test_threshold:]

        return dataset, train, test
    
    def create_tokenizer(self,lines):

        """Create a tokenizer based on input lines.

        Args:
            lines (list): List of strings to tokenize.

        Returns:
            Tokenizer: Created tokenizer.
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(lines)
        return tokenizer
 

    def encode_sequences(self,tokenizer, length, lines):

        """Encode sequences using the given tokenizer and pad them to a specified length.

        Args:
            tokenizer (Tokenizer): Tokenizer to use for encoding.
            length (int): Length to pad sequences to.
            lines (list): List of sequences to encode.

        Returns:
            numpy.ndarray: Encoded and padded sequences.
        """
      
        X = tokenizer.texts_to_sequences(lines) # integer encode sequences
        X = pad_sequences(X, maxlen=length, padding='post') # pad sequences with 0 values
        return X
    
    
    def encode_output(self,sequences, vocab_size):

        """One-hot encode target sequences.

        Args:
            sequences (numpy.ndarray): Sequences to encode.
            vocab_size (int): Size of the vocabulary.

        Returns:
            numpy.ndarray: One-hot encoded sequences.
        """

        ylist = list()
        for sequence in sequences:
            encoded = to_categorical(sequence, num_classes=vocab_size)
            ylist.append(encoded)
        y = np.array(ylist)
        y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
        return y
 

    def _convert_sentence_to_vectors(self):
         
        """Convert sentences to vector representations for training."""

        dataset, train, test= self. _generate_train_test_split()
        # Prepare target tokenizer
        tar_tokenizer = self.create_tokenizer(dataset[:, self.idx_tar]) #save
        tar_vocab_size = len(tar_tokenizer.word_index) + 1  #save
        tar_length = 15  #save
        print(f'\nTarget ({self.target_str}) Vocabulary Size: {tar_vocab_size}')
        print(f'Target ({self.target_str}) Max Length: {tar_length}')

        '''Prepare source tokenizer'''
        src_tokenizer = self.create_tokenizer(dataset[:, self.idx_src])  #save
        src_vocab_size = len(src_tokenizer.word_index) + 1  #save
        src_length = 15  

        '''Save Parameters'''
        self.tar_tokenizer=tar_tokenizer
        self.tar_vocab_size=tar_vocab_size
        self.src_tokenizer=src_tokenizer
        self.tar_length=tar_length
        self.src_vocab_size=src_vocab_size
        self.src_length=src_length
    
        print(f'\nSource ({self.source_str}) Vocabulary Size: {src_vocab_size}')
        print(f'Source ({self.source_str}) Max Length: {src_length}\n')

        '''PREPARING TRAINING DATA '''
        trainX = self.encode_sequences(src_tokenizer, src_length, train[:, self.idx_src])
        trainY = self.encode_sequences(tar_tokenizer, tar_length, train[:, self.idx_tar])
        trainY = self.encode_output(trainY, tar_vocab_size)
        self.trainX=trainX
        self.trainY=trainY
        pass
    
    def create_model(self, src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
      
        """Create the sequence-to-sequence model.

        Args:
            src_vocab (int): Source vocabulary size.
            tar_vocab (int): Target vocabulary size.
            src_timesteps (int): Length of source sequences.
            tar_timesteps (int): Length of target sequences.
            n_units (int): Number of units in the LSTM layer.

        Returns:
            Sequential: Created model.
        """

        model = Sequential()
        model.add(Embedding(src_vocab, n_units,  mask_zero=True))
        model.add(LSTM(n_units))
        model.add(RepeatVector(tar_timesteps))
        model.add(LSTM(n_units, return_sequences=True))
        model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
        return model
    
    def train(self):
        """Train the translation model."""

        self._convert_sentence_to_vectors()
     
        model=self.model
        model = self.create_model(self.src_vocab_size, self.tar_vocab_size, self.src_length, self.tar_length, 256)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
       
        history = model.fit(self.trainX, 
               self.trainY, 
                epochs=2 ,
                batch_size=64, 
                validation_split=0.1, 
                verbose=1,
                callbacks=[
                                EarlyStopping(
                                monitor='val_loss',
                                patience=10,
                                restore_best_weights=True
                            )
                    ])


        model.summary()
        self.model=model

        self.save_models_and_parameters(total_sentences=TRAINING_SIZE, model= model,src_tokenizer=self.src_tokenizer, tar_tokenizer=self.tar_tokenizer, src_length=self.src_length, tar_length=self.tar_length, src_vocab_size=self.src_vocab_size,tar_vocab_size=self.tar_vocab_size )
        pd.DataFrame(history.history).plot()
        plt.title("Loss")
        plt.show()
        pass

    def save_models_and_parameters( self, total_sentences ,model ,src_tokenizer , tar_tokenizer, src_length, tar_length, src_vocab_size,tar_vocab_size ):
       
        """Save the model and associated parameters in a designated folder.

            Args:
                total_sentences (int): Total number of sentences.
                model (keras.Model): Trained model to be saved.
                src_tokenizer (Tokenizer): Tokenizer for the source language.
                tar_tokenizer (Tokenizer): Tokenizer for the target language.
                src_length (int): Length of source sequences.
                tar_length (int): Length of target sequences.
                src_vocab_size (int): Size of the source vocabulary.
                tar_vocab_size (int): Size of the target vocabulary.
        """
       
        model_name =str(total_sentences)
        path='../temp_model/'+model_name+"/"
        
        src_parameters={
            'src_length': src_length,
            'src_vocab_size': src_vocab_size,
        }
        src_tokenizer= src_tokenizer

        target_parameters={
            'target_length': tar_length,
            'target_vocab_size': tar_vocab_size,
        }
        target_tokenizer= tar_tokenizer

        model.save(path+'lstm_model.h5' ) 
        with open(path+'src_parameters.pickle', 'wb') as handle:
            pickle.dump(src_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path+'src_tokenizer.pickle', 'wb') as handle:
            pickle.dump(src_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path+'target_parameters.pickle', 'wb') as handle:
            pickle.dump(target_parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path+'target_tokenizer.pickle', 'wb') as handle:
            pickle.dump(target_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def load_model(self,model_path ):
       
        """
        Load a saved model and associated parameters.

        Args:
            model_path (str): Path to the directory containing the saved model and parameters.
        """

        model, src_tokenizer, target_tokenizer, src_parameters, target_parameters= self._load_models_and_parameters(model_path)
        print(model, src_tokenizer, target_tokenizer, src_parameters, target_parameters)
       
        self.model=model
        
        self.src_tokenizer=src_tokenizer
        
        self.tar_tokenizer=target_tokenizer

        self.src_length=src_parameters["src_length"]
        self.src_vocab_size=src_parameters["src_vocab_size"]

        self.tar_length=target_parameters["target_length"]
        self.tar_vocab_size=target_parameters["target_vocab_size"]
        pass

    def _load_models_and_parameters(self, model_size):

        """
        Load the saved model and associated parameters from the specified directory.

        Args:
           model_size (str): Path to the directory containing the saved model and parameters.

        Returns:
            tuple: Tuple containing loaded model, source tokenizer, target tokenizer, source parameters, and target parameters.
        """

        path=model_size+'/'

        model = keras.models.load_model(path+'lstm_model')
        with open(path+ "src_parameters.pickle", 'rb') as handle:
            src_parameters = pickle.load(handle)

        with open(path+ "src_tokenizer.pickle", 'rb') as handle:
            src_tokenizer = pickle.load(handle)

        with open(path+ "target_parameters.pickle", 'rb') as handle:
            target_parameters = pickle.load(handle)

        with open(path+ "target_tokenizer.pickle", 'rb') as handle:
            target_tokenizer = pickle.load(handle)
        return model, src_tokenizer, target_tokenizer, src_parameters, target_parameters
    
    def word_for_id(self,integer, tokenizer):
        """
        Map an integer to a word using the provided tokenizer.

        Args:
            integer (int): Integer to be mapped to a word.
            tokenizer (Tokenizer): Tokenizer containing word-to-index mapping.

        Returns:
            str: Word corresponding to the integer index.
        """

     
        for word, index in tokenizer.word_index.items():
            if index == integer:
                return word
        return None
 
    def predict_seq(self, model, tokenizer, source):
       
        """Generate a target sequence from a source sequence using the provided model and tokenizer.

        Args:
            model (keras.Model): Trained model for sequence prediction.
            tokenizer (Tokenizer): Tokenizer for the target language.
            source (numpy.ndarray): Encoded source sequence.

        Returns:
            str: Predicted target sequence.
        """

        prediction = model.predict(source, verbose=0)[0]
        integers = [np.argmax(vector) for vector in prediction]
        target = list()
        for i in integers:
            word = self.word_for_id(i, tokenizer)
            if word is None:
                break
            target.append(word)
        return ' '.join(target)
    

    def compare_prediction(self,model, tar_tokenizer, sources, raw_dataset, limit=20):
        """
        Compare the predictions made by the model with the actual target sequences.

        Args:
            model (keras.Model): Trained model for sequence prediction.
            tar_tokenizer (Tokenizer): Tokenizer for the target language.
            sources (numpy.ndarray): Encoded source sequences.
            raw_dataset (list): List of raw source-target pairs.
            limit (int, optional): Maximum number of predictions to compare. Defaults to 20.

        Returns:
            tuple: Tuple containing actual and predicted sequences.
        """
       
        actual, predicted  = [], []
        src = f'{self.source_str.upper()} (SOURCE)'
        tgt = f'{self.target_str.upper()} (TARGET)'
        pred = f'AUTOMATIC TRANSLATION IN {self.target_str.upper()}'
        print(f'{src:30} {tgt:25} {pred}\n')
        
        for i, source in enumerate(sources): # translate encoded source text
            source = source.reshape((1, source.shape[0]))
            translation = self.predict_seq(model, tar_tokenizer, source)
            raw_src,raw_target = raw_dataset[i]
            print("##############################################################################")
            print(f' {i+1}. {raw_src:30} || {raw_target:25} || {translation}')
            ## STORE PREDICTIONS
            ############################################
            actual.append(raw_target.split())
            predicted.append(translation.split())
            ############################################
            if i >= limit: # Display some of the result
                break
        return actual, predicted
    
    def genereate_test_results(self):
        """Generate and compare predictions on a test dataset.

        Returns:
            tuple: Tuple containing actual and predicted sequences.
        """
        data_size=1000

        pool_oftexts, pairs =createDataset(data_size=data_size, type="train")
        dataset= pool_oftexts
        dataset = dataset.values
        test=dataset
        testX = self.encode_sequences(self.src_tokenizer, self.src_length, test[:, self.idx_src])
        testY = self.encode_sequences(self.tar_tokenizer, self.tar_length, test[:, self.idx_tar])
        testY = self.encode_output(testY, self.tar_vocab_size)
        actual, predicted= self.compare_prediction(self.model, self.tar_tokenizer, testX, test)
        return actual, predicted
    



if __name__ == "__main__":

    # SET THE VALUES BEFORE RUNNING
    TRAINING_SIZE=3000
    translator= Translator(training_size=TRAINING_SIZE)
    translator.train()