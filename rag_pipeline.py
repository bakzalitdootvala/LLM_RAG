import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# -----------------------------
# 1. Embeddings
# -----------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL)

# Example texts for vectorization
sentences = [
    "Large Language Models (LLMs) are transforming the world of AI.",
    "Retrieval-Augmented Generation (RAG) enables smarter applications.",
    "Python programming powers modern AI pipelines.",
    "Artificial Intelligence is revolutionizing technology."
]

embeddings = embedder.encode(sentences)
print("Embeddings ready. Shape:", embeddings.shape)

# -----------------------------
# 2. Local generative model
# -----------------------------
MODEL_NAME = "gpt2"  # lightweight local model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)

def generate_answer(prompt, max_new_tokens=100):
    """
    Generate answer using local GPT-2 and avoid repetitive junk.
    """
    full_prompt = f"{prompt}\nJoke:"
    result = text_generator(
        full_prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    text = result[0]['generated_text'].split("Joke:",1)[-1].strip()
    # Take only the first sentence to prevent repetition
    text = text.split(".")[0]
    return text

# -----------------------------
# 3. Simple RAG function
# -----------------------------
def ask_rag(question):
    """
    Finds the most similar embedding and builds a prompt for the model.
    """
    question_emb = embedder.encode([question])[0]

    # Cosine similarity
    similarities = np.dot(embeddings, question_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_emb)
    )
    best_idx = np.argmax(similarities)
    context = sentences[best_idx]

    prompt = f"Using the context: '{context}', answer the question: {question}"
    answer = generate_answer(prompt)
    return answer

# -----------------------------
# 4. Run example
# -----------------------------
if __name__ == "__main__":
    while True:
        question = input("\nAsk the model a question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
        response = ask_rag(question)
        print("\nModel answer:\n", response)
