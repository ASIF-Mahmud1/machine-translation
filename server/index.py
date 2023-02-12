from statistics import mode
import uvicorn
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from pyngrok import ngrok

from inference import predict_sentence
app = FastAPI()


class Sentence(BaseModel):
    sentence: str


@app.get('/')
def index():
    return {
        'message': 'This is the homepage of the Statistical Machine Translation ',
    }

# example: url/predict?sentence=%22make%20life%20good,%20dont%20get%20bitter%22


@app.get('/predict')
def get_translation(sentence: str):
    print('Query params ', sentence)
    result = predict_sentence(sentence)

    return {'prediction': result}


# NGROK URL
ngrok.connect(8000, )
tunnels = ngrok.get_tunnels()
print("NGROK URL\n", tunnels)
# NGROK URL

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
