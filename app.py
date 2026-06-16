import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

st.title("Embedding explorer")


SENTENCES = [
    # Group 1: Animals
    "The quick brown fox jumps over the lazy dog.",
    "A dark-colored fox leaps above a sleepy canine.",
    "A sly fox swiftly moves past a tired hound.",
    "Dogs are loyal companions and love to play fetch.",
    "Foxes and dogs are both members of the canine family.",

    # Group 2: Technology/AI
    "Artificial intelligence is transforming the world.",
    "Machine learning algorithms are advancing rapidly.",
    "Deep learning is a powerful branch of artificial intelligence.",
    "Robots powered by AI are becoming more common in industry.",
    "Natural language processing helps computers understand text.",

    # Group 3: Food
    "Pizza is my favorite food to eat on weekends.",
    "I love making homemade pizza with fresh ingredients.",
    "Burgers and fries are a classic fast food combination.",
    "Sushi is a popular Japanese dish made with rice and fish.",
    "Many people enjoy ice cream as a dessert in the summer."
]


EMBEDDING_MODEL = "/Users/nadak/models/all-MiniLM-L6-v2"

@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_data
def get_embeddings(sentences):
    model = load_model()
    embeddings = model.encode(sentences)
    return embeddings

st.write(f"Generated embeddings shape: {get_embeddings(SENTENCES).shape}") 