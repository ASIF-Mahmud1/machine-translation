import numpy as np

def str_to_tokens( sentence : str , encoder_word_dict, max_input_length, preprocessing):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
            # print("word ", word, eng_word_dict.get(word,1) )
            my_word=  encoder_word_dict.get(word,1)
            tokens_list.append(my_word) 

    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=max_input_length , padding='post')



def translate_sentence(sentence, enc_model,dec_model, encoder_word_dict,  decoder_word_dict , max_output_length, preprocessing ):
        for epoch in range(1 ):
            states_values = enc_model.predict( str_to_tokens(sentence, encoder_word_dict ,max_output_length, preprocessing ) )
            empty_target_seq = np.zeros( ( 1 , 1 ) )
            empty_target_seq[0, 0] = decoder_word_dict['start']
            stop_condition = False
            decoded_translation = ''
            while not stop_condition :
                dec_outputs , h , c = dec_model.predict([ empty_target_seq ] + states_values )
                sampled_word_index = np.argmax( dec_outputs[0, -1, :] )
                sampled_word = None
                for word , index in decoder_word_dict.items() :
                    if sampled_word_index == index :
                        decoded_translation += ' {}'.format( word )
                        sampled_word = word
                
                if sampled_word == 'end' or len(decoded_translation.split()) > max_output_length:
                    stop_condition = True
                    
                empty_target_seq = np.zeros( ( 1 , 1 ) )  
                empty_target_seq[ 0 , 0 ] = sampled_word_index
                states_values = [ h , c ] 

            # print("Decoded Traslation ", decoded_translation )
        return  decoded_translation
