import streamlit as st
print(">>> ÉTAPE 1 : Imports terminés")

st.title("Test de blocage")
print(">>> ÉTAPE 2 : Titre affiché")

from langchain_huggingface import HuggingFaceEmbeddings
print(">>> ÉTAPE 3 : Début chargement Embeddings")

# C'est ici que ça risque de bloquer
model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print(">>> ÉTAPE 4 : Embeddings chargés !")

st.success("Si tu vois ça, c'est que ça marche !")