# Episode 004 — Embedding explorer

> [One sentence single takeaway from this project.]

## The Problem / The Question
Is it possible to visualise what embeddings actually are?

## What I Built
Plain English description of what was implemented. What it does and how it works at a high level.

## What I Learned
- [Specific finding — not generic]
- [The thing that surprised me]
- [The assumption that broke]
- [The detail worth remembering]

## How to Run
Step-by-step instructions. 

## Tech Used
- [Huggingface](https://huggingface.co/models) -> Tasks -> Feature Extraction -> libraries -> sentence-transformers -> Sort by Most Downloads
> I choose [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) as my local free embeddings model this time. It is fast small (33MB) and english only, making it a practical choice. 
- sentence-transformers — downloads and runs the embedding model locally
- streamlit — turns a Python file into a web app, no HTML needed
- numpy — maths library, for the cosine similarity calculation

## References
- [Source 1]
- [Source 2]