from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None

    def build(self, documents):
        self.db = FAISS.from_documents(documents, self.embeddings)
        return self.db

    def get_retriever(self):
        if not self.db:
            raise ValueError("Vector store not built yet.")
        return self.db.as_retriever()
