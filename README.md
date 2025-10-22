LLM_RAG — Retrieval-Augmented Generation Playground

A minimal yet powerful example of Retrieval-Augmented Generation (RAG) using "SentenceTransformers" for embeddings and "ChromaDB" for vector storage.

This project demonstrates how to combine semantic search and generative AI to build intelligent, context-aware systems — all locally, no external APIs required.

Features:

1)Sentence Embeddings with "all-MiniLM-L6-v2"
2)ector Search using "ChromaDB"
3)Simple RAG Pipeline that connects retrieval + reasoning
4)Fully local, lightweight, and easy to extend


Installation:

Clone the repo and install dependencies:


git clone https://github.com/<your-username>/LLM_RAG.git,
cd LLM_RAG,
pip install -r requirements.txt


Future Ideas: integrate a local LLM (like Flan-T5 or Mistral) for generative responses, add document loaders and chunking for long text files, connect to a web or chatbot frontend

Tech Stack:
Python 3.10+,
SentenceTransformers,
ChromaDB,
(Optional) LangChain for advanced chaining logic.
