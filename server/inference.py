import numpy as np
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok
from tensorflow import keras
from google.transliteration import transliterate_word ## not installed
from tensorflow.keras.preprocessing.sequence import pad_sequences ## causing issue


model_size='15000'

def load_models_and_parameters(model_size):

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

model_path=    "../machine-learning/model/lstm/" +model_size
model, src_tokenizer, target_tokenizer, src_parameters, target_parameters= load_models_and_parameters(model_path)



src_length=src_parameters["src_length"]
src_vocab_size=src_parameters["src_vocab_size"]

target_length=target_parameters["target_length"]
target_vocab_size=target_parameters["target_vocab_size"]

print(src_length, target_length, src_vocab_size, target_vocab_size)

def encode_sequences(tokenizer, length, lines):  ## pass src_tokenizer for tokenizer
    # encode and pad sequences
    X = tokenizer.texts_to_sequences(lines) # integer encode sequences
    X = pad_sequences(X, maxlen=length, padding='post') # pad sequences with 0 values
    return X

def word_for_id(integer, tokenizer):
    # map an integer to a word
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 
def predict_seq(model, tokenizer, source):  ## pass target_tokenizer for tokenizer
    # generate target from a source sequence
    prediction = model.predict(source, verbose=0)[0]
    integers = [np.argmax(vector) for vector in prediction]
    target = list()
    for i in integers:
        word = word_for_id(i, tokenizer)
        if word is None:
            break
        target.append(word)
    return ' '.join(target)


transliterate_eng_hindi = transliterate_word('yah hamaare desh ke lie vaastav mein anivaary vastu hai.', lang_code='hi', max_suggestions=1)
print(transliterate_eng_hindi)

encoded_hindi = encode_sequences(src_tokenizer, src_length, transliterate_eng_hindi)


source= encoded_hindi
tar_tokenizer=target_tokenizer

translation = predict_seq(model, tar_tokenizer, source)