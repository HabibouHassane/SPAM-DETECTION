import os




import os
from flask import Flask, jsonify, request
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
# from keras.preprocessing.sequence import pad_sequences
from keras.utils import pad_sequences

from keras.preprocessing.text import Tokenizer
import numpy as np
import pandas as pd


df = pd.read_csv("df_clean.csv")
tokenizer = Tokenizer()
msg = df.MESSAGE.values.tolist()
for i in range(len(msg)):
    if isinstance(msg[i], float):
        msg[i] = "null"
tokenizer.fit_on_texts(msg)
#input_text = tokenizer.texts_to_sequences(msg)
max_length = 7719
#padded_input = pad_sequences(input_text,maxlen=max_length,padding='post')


app = Flask(__name__)



@app.route("/", methods = ["GET"])
def hello():
    return jsonify({"hello":"nasser"})


@app.before_first_request
def load():
    model_path = "my_model.h5"
    model = load_model(model_path, compile=False)
    return model

# Chargement du model
model = load()


def preprocess(text):
    input_text = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(input_text,maxlen=max_length,padding='post')
    return padded_input


@app.route("/predict", methods=["POST"])
def predict():
    
    # Récupérer les données JSON de la réponse
    response_json = request.json
    
     # Récupérer la valeur associée à la clé "key"
    
    texte = response_json.get("texte")
    


    #traitement du texte
    texte = preprocess(texte)

    # predictions
    x = int(np.round(model.predict(texte)[0]))
    if x==0:
        pred = "Not Spam"
    else:
        pred = "Spam"
    

    #rec = pred[0][0].tolist()

    return jsonify({"predictions" : pred})

if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8000)