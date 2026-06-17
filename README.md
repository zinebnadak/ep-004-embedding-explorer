# Episode 004 — Embedding explorer

> No single number in a vector means anything, the meaning lives in the relationship between them (cosine similarity score (-1) - 1)

## The Problem / The Question
Is it possible to visualise what embeddings actually are?

## What I Built
A tool that takes 15 sentences, turns each one into a vector of 384 numbers using a local embedding model, then computes how similar every sentence is to every other sentence. The result is a colour-coded grid where you can see which sentences are close in meaning (green) and which aren't (red)

## What I Learned
- An embedding model turns text into a vector, but the quality of that vector depends on what you feed it. Single words vs short sentences on the heatmap looked completely different. A sentence gives the model enough context to be specific, while single words are too vague. That's why chunk size matters in RAG, the more contxt the better accuracy during retrieval.
- Sentence Transformers is a library built by HuggingFace that makes it easy to generate embeddings from text or images using pretrained models. It wraps the complexity so you can go from sentence to vector in one line.
- ALL_CAPS variable names are a Python convention meaning "this value doesn't change.Just a signal to anyone reading your code.

### Streamlit
- @st.cache_... is a decorator you put before a function. Streamlit runs it once and reuses the result forever. Without it, your model for example reloads on every page interaction.
- @st.cache_resource — for resources. Database connections, loaded models, file handles. Things that are heavy to create and should exist once.
- @st.cache_data — for data. Numbers, strings, lists, arrays. Things that can be copied and stored. The difference matters: a model is a resource, the embeddings it produces are data.
- st.write() — using it as a debugging tool in Streamlit. Same job as print() in regular Python. Dump anything you want to inspect directly onto the page.
- unsafe_allow_html=True — lets you write raw HTML inside a Streamlit markdown string rendered directly. That's what I used to colour the heatmap cells.


## How to Run
Clone the repo:
````
https://github.com/zinebnadak/ep-004-embedding-explorer.git
````

1. Run a quick check to see if uv is installed on the system, otherwise install it [here](https://docs.astral.sh/uv/getting-started/installation/) 

```
uv --version
```

2. Create a virtual environment with uv and activate it:
```
uv venv
```
```
source .venv/bin/activate
```

3. Install packages (equivalent of pip install -r requirements.txt) for uv its:
```
uv sync
```

To add additional packages later use:
````
uv add <package>
````

Check dowloads with:
````
uv pip list
````

4. Set up HF access token on your account. Type: Read. Copy token.

5. Download model. There are different ways to download a model from HF. I will be using HF CLI and download locally to a folder "models" on my machine entering the token directly in terminal. Update the EMBEDDING_MODEL variable in app.py, if necissary, to point to wherever you choose to downloaded the model on your machine. Paste the token to its place in the prompt below: 
````
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ~/models/all-MiniLM-L6-v2 --token YOUR_TOKEN_HERE
````

Check HuggingFace token is working. If this prints your HuggingFace username, your token is working. If it errors, check the token and try again before downloading the model. 
```
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='YOUR_TOKEN_HERE')
print(api.whoami()['name'])
"
```

6. Run app:
````
streamlit run app.py
````



## Tech Used
> Model: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 22.7M parameters, 80MB, runs fully offline. Maps each sentence to a 384-dimensional vector. Quite small, making it a practical choice. Found on [Huggingface](https://huggingface.co/models) -> Tasks -> Feature Extraction -> libraries -> sentence-transformers -> Sort by Most Downloads
- sentence-transformers — downloads and runs the embedding model locally
- numpy — math library, cosine similarity written from scratch
- streamlit — turns a Python file into a web app, no HTML needed

## Tests 
### Each row is a sentance with 384 vector embeddings. No comparison or similarity scores displayed yet.
![embeddings as arrays](images/pic1.png)

### For every sentence against every other sentence gives a comparison score. 20 sentences = 400 cells. The matrix is symmetric so that is why we se see numbers appear twice. [0][1] and [1][0] are always the same number
![raw similarity matrix](images/pic3.png)

### The colour math: score 1.0 (red=0, green=255 (bright green)) and score 0.0 (red=255, green=0 (red))
![color similarity matrix](images/pic4.png)

### The pair explorer takes a index (displayed as formatted sentences) and compares them to eachother resulting in a similarity score
![pair explorer](images/pic5.png)

### Top 5 most similar pairs and their scores
![top5](images/pic6.png)

## References
- [sentence generator](https://randomwordgenerator.com/sentence.php)
- [Creating and Visualizing Embeddings with Sentence Transformers | TensorTeach](https://www.youtube.com/watch?v=5S2Yk45xMLM)
- [Numpy - .zeros method: creates an empty matrix of zeros](https://numpy.org/devdocs/reference/generated/numpy.zeros.html)
- [Streamlit docs](https://docs.streamlit.io/get-started)