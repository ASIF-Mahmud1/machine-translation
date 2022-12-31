import tensorflow as tf
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

