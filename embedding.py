from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5"
        )

    def get(self):
        return self.model
