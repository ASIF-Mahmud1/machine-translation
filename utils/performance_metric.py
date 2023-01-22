import numpy as np
from statistics import mean
from scoreTest import get_cosine_val
from nltk.translate.bleu_score import corpus_bleu
import tensorflow as tf
import tensorflow_text as text
 
def word_for_id(integer, tokenizer):
    # map an integer to a word
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 
def predict_seq(model, tokenizer, source):
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
    
    
def create_dataframe_to_score(model, tar_tokenizer, sources, raw_dataset):
    # Get the bleu score of a model
    actual, predicted, actual_rouge , cosine_value_list= [], [],[], []
    for i, source in enumerate(sources):
        # translate encoded source text
        source = source.reshape((1, source.shape[0]))
        translation = predict_seq(model, tar_tokenizer, source)
        raw_src, raw_target = raw_dataset[i]
        actual.append([raw_target.split()])
        actual_rouge.append(raw_target.split())
        predicted.append(translation.split())
        #######################################################################
                    ####   Calculate Cosine Value   ####

        cosine_value= get_cosine_val (translation,raw_target )
        cosine_value_list.append(cosine_value)
        #######################################################################
        # print("predicted ",translation.split())    
   
    average_cosine= mean(cosine_value_list)
    return actual, predicted, actual_rouge, average_cosine


def bleu_score(actual, predicted):
    # Get the bleu score of a model
    bleu_dic = {}
    bleu_dic['bleu-1-grams'] = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
    bleu_dic['bleu-1-2-grams'] = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
    bleu_dic['bleu-1-3-grams'] = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0))
    bleu_dic['bleu-1-4-grams'] = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu_dic

def calculate_ROUGE(actual, predicted):
    hypotheses = tf.ragged.constant(predicted)
    references = tf.ragged.constant(actual)

    rouge_test= text.metrics.rouge_l(hypotheses, references)
    f_measure_list= rouge_test.f_measure
    p_measure_list= rouge_test.p_measure
    r_measure_list= rouge_test.r_measure
  

    f_measure_average= (f_measure_list.numpy()).mean()
    p_measure_average= (p_measure_list.numpy()).mean()
    r_measure_average=(r_measure_list.numpy()).mean()
    return { "f_measure_average":f_measure_average, "p_measure_average":p_measure_average, "r_measure_average":r_measure_average  }  




