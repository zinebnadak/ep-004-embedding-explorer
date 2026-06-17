import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

SENTENCES = [
    "I love pizza.",
    "I baked fresh bread today.",
    "Food is my favorite topic.",
    "The dog ran across the park.",
    "My cat sleeps all day.",
    "Animals are fascinating creatures.",
    "The robot completed the task.",
    "Artificial intelligence is changing everything.",
    "Computers process data very fast.",
    "We had a delicious meal tonight.",
    "Cars are getting more expensive.",
    "BMW makes luxury vehicles.",
    "Mercedes is a premium car brand.",
    "Audi is known for its design.",
    "The kitten played with a ball of yarn.",
]


EMBEDDING_MODEL = "/Users/nadak/models/all-MiniLM-L6-v2"


@st.cache_resource
def load_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_data
def get_embeddings(sentences):
    model = load_model()
    embeddings = model.encode(sentences, normalize_embeddings=True) 
    return embeddings

def cosine_similarity(vec_a, vec_b):
    dot = np.dot(vec_a, vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    return dot / (magnitude_a * magnitude_b) 

def build_similarity_matrix(embeddings):
    n = len(embeddings)
    matrix = np.zeros((n, n))
    for i in range (n):
        for j in range (n):
            matrix[i][j] = cosine_similarity(embeddings[i], embeddings[j])
    return matrix 

def main():
    st.set_page_config(page_title="Embedding Explorer", layout="wide")
    st.title("Embedding Explorer")
    with st.sidebar:
        st.header("Sentences")
        sentences = []
        for index, sentence in enumerate(SENTENCES):
            edited = st.text_input(f"{index+1}.", value = sentence, key = f"sent_{index}")
            sentences.append(edited)
    embeddings = get_embeddings(tuple(sentences))
    matrix = build_similarity_matrix(embeddings)

    st.subheader("Similarity Matrix")

    n = len(sentences)

    for i in range(n):
        cols = st.columns(n+1)
        cols[0].markdown(f"{i+1}.")
        for j in range(n):
            score = matrix[i][j]

            red = int(255*(1-score))
            green = int(255*score)
            color = f"rgb({red},{green},80)"

            cols[j+1].markdown(f'<div style="background:{color};padding:4px;text-align:center;'f'font-size:10px;">{score:.2f}</div>', unsafe_allow_html=True)

    st.subheader("Pair Explorer")
    index_a = st.selectbox("Sentence A", range(n),
                           format_func= lambda i: sentences[i])
    index_b = st.selectbox("Sentence B", range(n), index=1,
                           format_func= lambda i: sentences[i])
    
    score = matrix[index_a][index_b]
    st.metric("Similarity", f"{score}")

    st.subheader("Most Similar Pairs")
    pairs = []
    for i in range(n):
        for j in range(i+1,n): #start j above so it do not repeat pairs
            pairs.append((matrix[i][j], i, j))

    pairs.sort(reverse=True) #highest score first

    for score, i, j in pairs[:5]:
        st.write(f"{score} - {sentences[i]} <-> {sentences[j]}")


if __name__ == "__main__":
    main()

 