import os
import logging
from extract import PDFLoader
from chunks import Chunker
from embedding import EmbeddingModel
from vector import VectorStoreManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def build_index():
    logging.info("🚀 Starting FAISS index building process...")

    # STEP 1: Load documents
    loader = PDFLoader("medical_book.pdf")
    logging.info(f"📂 Loading documents from {loader.path}...")
    documents = loader.load()
    if not documents:
        raise ValueError(f"No documents loaded from {loader.path}")
    logging.info(f"✅ Loaded {len(documents)} documents")

    # STEP 2: Chunking
    chunker = Chunker()
    logging.info("✂️ Splitting documents into chunks...")
    chunks = chunker.chunk(documents)
    if not chunks:
        raise ValueError("No chunks generated from documents")
    logging.info(f"✅ Generated {len(chunks)} chunks")

    # STEP 3: Embeddings
    logging.info("🧠 Generating embeddings...")
    embeddings = EmbeddingModel().get()
    logging.info("✅ Embeddings model loaded successfully")

    # STEP 4: Vector Store & Save
    logging.info("💾 Building FAISS vector store...")
    vector_store = VectorStoreManager(embeddings)
    db = vector_store.build(chunks)

    save_path = "faiss_index"
    os.makedirs(save_path, exist_ok=True)
    db.save_local(save_path)
    logging.info(f"✅ FAISS index built and saved at {save_path}")

if __name__ == "__main__":
    build_index()
