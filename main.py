from extract import PDFLoader
from chunks import Chunker
from embedding import EmbeddingModel
from vector import VectorStoreManager

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub


def main():
    # STEP 1: Load documents
    loader = PDFLoader("medical_book.pdf")
    documents = loader.load()
    if not documents:
        raise ValueError(f"No documents loaded from {loader.path}")

    # STEP 2: Chunking
    chunker = Chunker()
    chunks = chunker.chunk(documents)
    if not chunks:
        raise ValueError("No chunks generated from documents")

    # STEP 3: Embeddings
    embeddings = EmbeddingModel().get()

    # STEP 4: Vector Store
    vector_store = VectorStoreManager(embeddings)
    db = vector_store.build(chunks)
    retriever = vector_store.get_retriever()

    # STEP 5: LLM (Mistral)
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )

    # STEP 6: Retrieval QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    print("\n🤖 RAG Chatbot is ready! Type 'exit' to quit.\n")

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
