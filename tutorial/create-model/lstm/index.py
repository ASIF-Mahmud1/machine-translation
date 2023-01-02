import numpy as np
import tensorflow as tf
from tensorflow.keras import layers , activations , models , preprocessing , utils
import pandas as pd
from tensorflow.keras.layers import Input, Embedding,Dense,  LSTM
from tensorflow.keras.models import load_model
import string
import pickle
from tensorflow import keras
from keras.utils.vis_utils import plot_model
import statistics
import math
import os
import sys

def handle_system_path():
    preprocessing_path=os.path.abspath(os.path.join('/Users/learn/Desktop/Projects/machine-translation/tutorial/utils')) 
    sys.path.append(preprocessing_path)      
    print(sys.path)

handle_system_path()

def createInputDataForEncoder(lines):
    hindi_lines = list()
    for line in lines.hindi:
        hindi_lines.append( line ) 
        
    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts( hindi_lines ) 
    tokenized_hindi_lines = tokenizer.texts_to_sequences( hindi_lines ) 
    length_list = list()

    for token_seq in tokenized_hindi_lines:
        length_list.append( len( token_seq ))

    # max_input_length = np.array( length_list ).max()
    max_input_length = math.floor(statistics.mode(length_list))
    print( 'Hindi max length is {}'.format( max_input_length ))

    padded_hindi_lines = preprocessing.sequence.pad_sequences( tokenized_hindi_lines , maxlen=max_input_length , padding='post' )
    encoder_input_data = np.array( padded_hindi_lines )
   
    hindi_word_dict = tokenizer.word_index
    num_hindi_tokens = len( hindi_word_dict )+1
    # print("Hindi Dictionary" ,hindi_word_dict  )
    return max_input_length, num_hindi_tokens, encoder_input_data, hindi_word_dict
   

def createInputDataForDecoder(lines):
    eng_lines = list()
    for line in lines.eng:
        eng_lines.append( '<START> ' + line + ' <END>' )  

    tokenizer = preprocessing.text.Tokenizer(oov_token=1)
    tokenizer.fit_on_texts( eng_lines ) 
    tokenized_eng_lines = tokenizer.texts_to_sequences( eng_lines ) 

    length_list = list()
    for token_seq in tokenized_eng_lines:
        length_list.append( len( token_seq ))
   
    max_output_length = math.floor(statistics.mode(length_list))
    print( 'English max length is {}'.format( max_output_length ))

    padded_eng_lines = preprocessing.sequence.pad_sequences( tokenized_eng_lines , maxlen=max_output_length, padding='post' )
    decoder_input_data = np.array( padded_eng_lines  )
    print( 'Decoder input data shape -> {}'.format( decoder_input_data.shape ))

    eng_word_dict = tokenizer.word_index
    num_eng_tokens = len( eng_word_dict )+1
    print( 'Number of English tokens = {}'.format( num_eng_tokens))

    return max_output_length, num_eng_tokens, decoder_input_data, eng_word_dict, tokenized_eng_lines


def createDecoderTargetData(tokenized_eng_lines, max_output_length, num_eng_tokens):
    decoder_target_data = list()
    for token_seq in tokenized_eng_lines:
        decoder_target_data.append( token_seq[ 1 : ] ) 
        
    padded_eng_lines = preprocessing.sequence.pad_sequences( decoder_target_data , maxlen=max_output_length, padding='post' )
    onehot_eng_lines = utils.to_categorical( padded_eng_lines , num_eng_tokens )
    decoder_target_data = np.array( onehot_eng_lines )
    print( 'Decoder target data shape -> {}'.format( decoder_target_data.shape ))
    return decoder_target_data 



def create_encoder_decoder(encoder_inputs,encoder_states,decoder_lstm,decoder_embedding,decoder_dense,decoder_inputs):
    
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    decoder_state_input_h = tf.keras.layers.Input(shape=( 256,))
    decoder_state_input_c = tf.keras.layers.Input(shape=( 256 ,))
    
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding , initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model 


def createEncoderDecoderModel(max_input_length, max_output_length, num_hindi_tokens, num_eng_tokens):
    encoder_inputs = Input(shape=( max_input_length ,  ))
    encoder_embedding = Embedding( num_hindi_tokens, 256 , mask_zero=True ) (encoder_inputs)
    encoder_outputs , state_h , state_c = LSTM( 256 , return_state=True , recurrent_dropout=0.2 , dropout=0.2 )( encoder_embedding )
    encoder_states = [ state_h , state_c ]

    decoder_inputs = Input(shape=( max_output_length , ))
    decoder_embedding = Embedding( num_eng_tokens, 256 , mask_zero=True) (decoder_inputs)
    decoder_lstm = LSTM( 256 , return_state=True , return_sequences=True , recurrent_dropout=0.2 , dropout=0.2)
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = Dense( num_eng_tokens , activation=tf.keras.activations.softmax ) 
    output = decoder_dense ( decoder_outputs )

    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy')
    encoder_model , decoder_model= create_encoder_decoder(encoder_inputs,encoder_states,decoder_lstm,decoder_embedding,decoder_dense,decoder_inputs)
    return model ,encoder_model , decoder_model



def create_LSTM_Model(lines):
    max_input_length, num_hindi_tokens, encoder_input_data, input_dict= createInputDataForEncoder(lines=lines)
    max_output_length, num_eng_tokens, decoder_input_data, eng_word_dict, tokenized_eng_lines =createInputDataForDecoder(lines=lines)
    decoder_target_data= createDecoderTargetData(tokenized_eng_lines, max_output_length, num_eng_tokens)
    model, encoder_model , decoder_model= createEncoderDecoderModel(max_input_length, max_output_length, num_hindi_tokens, num_eng_tokens)
    return  model, encoder_model , decoder_model, decoder_target_data, encoder_input_data, input_dict, decoder_input_data, eng_word_dict,  max_input_length,num_hindi_tokens, max_output_length, num_eng_tokens

  