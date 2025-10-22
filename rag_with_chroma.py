from sentence_transformers import SentenceTransformer
import chromadb

#initialization
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
sentences = [
    "Large Language Models (LLMs) are transforming the world of AI.",
    "Retrieval-Augmented Generation (RAG) enables smarter applications.",
    "Python programming powers modern AI pipelines.",
    "Artificial Intelligence is revolutionizing technology."
]
embeddings = embedder.encode(sentences).tolist()

client = chromadb.Client()

#collection
collection_name = "rag_collection"
if collection_name in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

collection.add(
    documents=sentences,
    metadatas=[{"source": f"sentence_{i}"} for i in range(len(sentences))],
    ids=[str(i) for i in range(len(sentences))],
    embeddings=embeddings 
)

#search
query = "What enables smarter AI applications?"
query_emb = embedder.encode([query]).tolist()  
results = collection.query(
    query_embeddings=query_emb,
    n_results=1
)
print("Query results:", results['documents'])
