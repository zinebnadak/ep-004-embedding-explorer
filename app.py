import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

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
    cols = st.columns(n+1)

    for i in range(n):
        for j in range(n):
            score = matrix[i][j]

            red = int(255*(1-score))
            green = int(255*score)
            color = f"rgb({red},{green},80)"

            cols[j+1].markdown(f'<div style="background:{color};padding:4px;text-align:center;'f'font-size:10px;">{score:.2f}</div>', unsafe_allow_html=True)

    st.subheader("Pair Explorer")
    index_a = st.selectbox("Sentance A", range(n),
                           format_func= lambda i: sentences[i])
    index_b = st.selectbox("Sentance B", range(n), index=1,
                           format_func= lambda i: sentences[i])
    
    score = matrix[index_a][index_b]
    st.metric("Similarity", f"{score}")

if __name__ == "__main__":
    main()
