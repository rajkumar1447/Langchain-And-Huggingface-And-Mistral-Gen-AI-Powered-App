from embedding import EmbeddingModel
from vector import VectorStoreManager
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import os


def main():
    save_path = "faiss_index"

    # STEP 1: Load embeddings
    embeddings = EmbeddingModel().get()

    # STEP 2: Load FAISS index if available
    if not os.path.exists(save_path):
        raise ValueError("FAISS index not found. Please run build_index.py first.")

    print(f"🔄 Loading FAISS index from {save_path}...")
    vector_store = VectorStoreManager(embeddings)
    db = vector_store.load(save_path)
    retriever = db.as_retriever()

    # STEP 3: LLM (Ollama Mistral)
    llm = Ollama(
        model="mistral",  # or "mistral:latest" if you pulled that
        temperature=0.7,
    )

    # STEP 4: Retrieval QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    print("\n🤖 RAG Chatbot is ready with Ollama Mistral! Type 'exit' to quit.\n")

    # Interactive loop
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        response = qa_chain.run(query)
        print("Bot:", response)


if __name__ == "__main__":
    main()
