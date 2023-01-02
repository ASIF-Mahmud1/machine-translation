from datasets import load_dataset
from numpy import array
import os
import sys
import pandas as pd

sys.path.append('/Users/learn/Desktop/Projects/machine-translation/utils')                                       


from preprocessing_text import removePunctuation, toLowercase

dataset = load_dataset("cfilt/iitb-english-hindi")

def createDataset( data_size,type):
    pairs=[] 
    for translation_pair in dataset[type]["translation"][:data_size]:
        source_sentence = translation_pair["hi"]
        target_sentence = translation_pair["en"]
        pairs.append([source_sentence, target_sentence])
    pairs = array(pairs)
    pairs= toLowercase(pairs)
    pairs=removePunctuation(pairs)

    lines= pd.DataFrame(columns=[ "hindi","eng"], data=pairs)
    lines= lines[:data_size]
   
    return lines