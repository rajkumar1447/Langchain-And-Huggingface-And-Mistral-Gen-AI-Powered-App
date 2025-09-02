from langchain_community.vectorstores import FAISS

class VectorStoreManager:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.db = None

    def build(self, chunks):
        self.db = FAISS.from_documents(chunks, self.embeddings)
        return self.db

    def get_retriever(self):
        return self.db.as_retriever()

    def save(self, path: str):
        if self.db:
            self.db.save_local(path)

    def load(self, path: str):
        self.db = FAISS.load_local(path, self.embeddings, allow_dangerous_deserialization=True)
        return self.db
