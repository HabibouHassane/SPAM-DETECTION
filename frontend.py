import streamlit as st

import requests




st.title("Spam Classification")

texte = st.text_input("Rwite the text to classify")

c1, c2 = st.columns(2)

if texte:
    texte_json = {"texte":texte}

    req = requests.post("http://192.168.0.13:8000/predict", json=texte_json)
    resultat = req.json()
    rec = resultat["predictions"]
    

    c1.write(texte)
    c2.write(rec)
    