# LLM_RAG: Retrieval-Augmented Generation Playground 

Welcome to **LLM_RAG**, a badass little demo of Retrieval-Augmented Generation (RAG) that shows how to mix semantic search with generative AI to build a system that’s smarter than your average chatbot — and it runs on your potato laptop, no cloud bullshit required!  This project is my way of saying, “Yo, I get how RAG works,” without needing a supercomputer or a fat wallet.

## What’s This All About? 
This project is a minimal, local RAG pipeline. It takes a question, digs up the most relevant context from a small dataset using vector search, and generates an answer with a lightweight language model. No external APIs, no subscriptions, just pure Python and open-source tools. It’s like building a mini-JARVIS on a budget.

### How It Works
1. **Text to Vectors**: I use `sentence-transformers/all-MiniLM-L6-v2` to turn texts into 384-dimensional embeddings. It’s fast, lightweight, and doesn’t eat your RAM for breakfast.
2. **Vector Search**: Embeddings are stored in `ChromaDB`, a nifty vector database that finds the closest match to your question using cosine similarity.
3. **Answer Generation**: The best context is fed into `GPT-2` (yeah, it’s old-school, but my CPU can’t handle the cool kids like LLaMA). GPT-2 generates a response based on the context and question.
4. **All Local**: Everything runs on your machine. No cloud, no API keys, no “please pay $20/month for AI.”

## Features 
- **Sentence Embeddings**: Using `all-MiniLM-L6-v2` for compact, high-quality vector representations.
- **Vector Search**: `ChromaDB` handles fast and simple similarity search.
- **RAG Pipeline**: Combines retrieval and generation in a clean, extensible way.
- **Lightweight & Local**: Runs on modest hardware (think 8GB RAM, CPU-only).
- **Easy to Hack**: Small codebase, ready for you to mess with and expand.

## Why This Stack? 
- **SentenceTransformers (`all-MiniLM-L6-v2`)**: Chosen for its speed and low resource usage. It’s perfect for a demo on a weak machine, and it still delivers solid embeddings.
- **ChromaDB**: A lightweight vector store that’s easy to set up and doesn’t require a PhD to use.
- **GPT-2**: Yeah, it’s not the shiniest toy in 2025, but it’s small enough to run on my budget laptop without melting it. If I had a beefy GPU, I’d slap in Mistral or Flan-T5.
- **Python 3.10+**: Modern, clean, and no legacy crap.

The dataset is tiny (just 4 sentences for now) to keep things simple and demo-friendly. In a real project, you’d throw in Wikipedia articles, PDFs, or whatever else you want to search through.
## Installation:

# Clone the repo and install dependencies:


git clone https://github.com/<your-username>/LLM_RAG.git,
cd LLM_RAG,
python -m venv venv
source venv/bin/activate,
pip install -r requirements.txt,
python rag.py


## Future Ideas
Integrate a local LLM (like Flan-T5 or Mistral) for generative responses, add document loaders and chunking for long text files, connect to a web or chatbot frontend

## Tech Stack
Python 3.10+,
SentenceTransformers,
ChromaDB,
(Optional) LangChain for advanced chaining logic.

I built this to show I understand RAG: how to encode texts, search for context, and generate answers. It’s a proof-of-concept, not a production app, so don’t expect it to write your thesis (yet). My hardware’s limited (CPU-only, 8GB RAM), so I went with lightweight tools to make it work. If you’re hiring, hit me up — I can scale this shit up with better gear!
