from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader

class PDFLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self):
        if self.path.endswith(".pdf"):
            loader = PyPDFLoader(self.path)   # single PDF
        else:
            loader = PyPDFDirectoryLoader(self.path)  # folder of PDFs
        return loader.load()