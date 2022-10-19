import math
import re
from collections import Counter

WORD = re.compile(r"\w+")

def get_cosine(vec1, vec2):
    
    # get intersecting keys
    intersection = set(vec1.keys()) & set(vec2.keys())
    # multiply and sum weights
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    # compute denominator
    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    # return cosine score
    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator

def text_to_vector(text):
    words = WORD.findall(text)
    return Counter(words)
def get_cosine_val(text1,text2):
    
    # turn text into vector counts
    vector1 = text_to_vector(text1)
    vector2 = text_to_vector(text2)
    # compute similarity
    cosine = get_cosine(vector1, vector2)
    return cosine

def get_BLEU_score(df, sentence_bleu):
     ## Calculate BLEU score
    scores=[]
    for reference, candidate in zip(df['reference'], df['candidate']):
        list_of_references= [reference]  # we have only one reference sentence for each candidate sentence
        result= sentence_bleu(list_of_references, candidate,weights=(1, 0, 0, 0))  ## Check this line
        scores.append(result)
    return scores

def get_ROUGE_score(df,pd, tf,text):
    scores=[]
    for reference, candidate in zip(df['reference'], df['candidate']):
        temp =['captain', 'of', 'the', 'delta', 'flight']
        references =tf.ragged.constant([reference])
        hypotheses= tf.ragged.constant([candidate])

        result= text.metrics.rouge_l(hypotheses, references)
        
        result_str= " F-measure: "+str(result.f_measure.numpy()[0]) +"  Precision: "+str(result.p_measure.numpy()[0])+"  Recall: "+str(result.r_measure.numpy()[0])
        column=["f_measure", "p_measure", "r_measure"]
        data= [[result.f_measure.numpy()[0] ,result.p_measure.numpy()[0] , result.r_measure.numpy()[0] ]]
        metric= pd.DataFrame(data=data, columns=column)
        resultObj= {"f_measure": result.f_measure.numpy()[0] , "p_measure": result.p_measure.numpy()[0],  "r_measure":result.r_measure.numpy()[0] }  
        scores.append(resultObj)
    

    return scores  ## ROUGE score calculated
        



