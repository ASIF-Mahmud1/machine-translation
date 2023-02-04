import dill as pickle

def read_sents(filename):
    sents = []
    c=0
    with open(filename,'r') as fi:
        for li in fi:
            sents.append(li.split())
    return sents

with open('./ml-model/ibm_smt.pkl', "rb") as f:
    ibm_model = pickle.load(f)



fr_sent ="Il nous faut inverser la recette, pour découvrir ce dont sont vraiment capables nos cerveaux.".split()
tr_sent = []
for w in fr_sent:
    probs = ibm_model.translation_table[w]
    if(len(probs)==0):
        continue
    sorted_words = sorted([(k,v) for k, v in probs.items()],key=lambda x: x[1], reverse=True)
    top_word = sorted_words[1][0]
    if top_word is not None:
        tr_sent.append(top_word)



print("French sentence: ", " ".join(fr_sent))
print("Translated Eng sentence: ", " ".join(tr_sent))