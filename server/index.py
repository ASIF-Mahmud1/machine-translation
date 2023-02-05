from lib2to3.pytree import Base
from statistics import mode
import uvicorn
import pickle
from fastapi import FastAPI

from pydantic import BaseModel
from pyngrok import ngrok

app = FastAPI()



class Sentence(BaseModel):
    sentence: str

with open("./ml-model/ibm_smt.pkl", "rb") as f:
    ibm_model = pickle.load(f)


@app.get('/')
def index():
    return {'message': 'This is the homepage of the Statistical Machine Translation '}



@app.post('/predictSentence')
def get_translation(data:Sentence):
    print(type(data))
    fr_sent =data.sentence.split()
    tr_sent = []
    for w in fr_sent:
        probs = ibm_model.translation_table[w]
        if(len(probs)==0):
            continue
        sorted_words = sorted([(k,v) for k, v in probs.items()],key=lambda x: x[1], reverse=True)
        top_word = sorted_words[1][0]
        if top_word is not None:
            tr_sent.append(top_word)

    result=" ".join(tr_sent)
    return {'prediction': result}
    
# NGROK URL    
ngrok.connect(8000, )
tunnels = ngrok.get_tunnels()
print("NGROK URL\n",tunnels)
# NGROK URL    

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)