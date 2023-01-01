import tensorflow as tf
import pickle
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Input, Embedding,Dense,  LSTM

def make_inference_models(encoder_inputs, encoder_states, decoder_inputs,decoder_embedding, decoder_lstm, decoder_dense):
    
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


def get_reconstructed_model(model_path):
    reconstructed_model = keras.models.load_model(model_path)
    plot_model(reconstructed_model, to_file='modelsummary.png', show_shapes=True, show_layer_names=True)
    reconstructed_model.summary()


    ## Load Dictionaries and Parameters 
    path_encoder_parameters= path_encoder_parameters
    path_encoder_dictionary= path_encoder_dictionary
    path_decoder_parameters= path_decoder_parameters
    path_decoder_dictionary= path_decoder_dictionary

    # loading
    with open(path_encoder_parameters, 'rb') as handle:
        encoder_parameters = pickle.load(handle)

    # loading
    with open(path_encoder_dictionary, 'rb') as handle:
        encoder_dictionary = pickle.load(handle)

    # loading
    with open(path_decoder_parameters, 'rb') as handle:
        decoder_parameters= pickle.load(handle)

    # loading
    with open(path_decoder_dictionary, 'rb') as handle:
        decoder_dictionary = pickle.load(handle)    

    
    return reconstructed_model, encoder_parameters ,decoder_parameters , encoder_dictionary , decoder_dictionary


def generate_encoder_decoder_input_states(model_path):
    reconstructed_model, encoder_parameters ,decoder_parameters , encoder_dictionary , decoder_dictionary = get_reconstructed_model(model_path)
    encoder_inputs = reconstructed_model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = reconstructed_model.layers[4].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = keras.Model(encoder_inputs, encoder_states)
    latent_dim = 256  # Note: may be need to save in drive as well


    num_decoder_tokens =decoder_parameters['num_decoder_tokens']
    max_output_length= decoder_parameters['max_decoder_seq_length']
    max_input_length= encoder_parameters['max_encoder_seq_length']

    encoder_word_dict=encoder_dictionary
    decoder_word_dict= decoder_dictionary


    decoder_inputs = Input(shape=( max_output_length , ))
    decoder_embedding = Embedding( num_decoder_tokens, 256 , mask_zero=True) (decoder_inputs)

    decoder_lstm = LSTM( 256 , return_state=True , return_sequences=True , recurrent_dropout=0.2 , dropout=0.2)
    decoder_dense = Dense( num_decoder_tokens , activation=tf.keras.activations.softmax ) 

    return encoder_inputs, encoder_states, decoder_inputs,decoder_embedding, decoder_lstm, decoder_dense , encoder_word_dict , decoder_word_dict

