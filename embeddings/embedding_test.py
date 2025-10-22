from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)

sentences = ["Hi", "How are you?"]
embeddings = model.encode(sentences)
print(embeddings)
