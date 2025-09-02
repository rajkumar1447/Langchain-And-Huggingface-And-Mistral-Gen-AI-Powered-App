# Langchain-And-Huggingface-And-Mistral-Gen-AI-Powered-App

📂 Project Structure
Langchain-And-Huggingface-And-Mistral-Gen-AI-Powered-App/
│── main.py                # Main entry point
│── embedding.py           # Embedding model (HuggingFace)
│── extract.py             # PDF loading
│── chunks.py              # Text chunking
│── vector.py              # Vector store management (FAISS)
│── requirements.txt       # Project dependencies
│── README.md              # Documentation (this file)
│── medical_book.pdf/      # Your knowledge source (PDFs)

⚙️ Setup Instructions
1️⃣ Create and activate a virtual environment
# Create venv (Windows)
python -m venv venv

# Activate venv (Windows PowerShell)
.\venv\Scripts\activate

# Activate venv (Linux/Mac)
source venv/bin/activate

2️⃣ Install dependencies

Make sure pip is updated:

pip install --upgrade pip


Then install all required packages:

pip install -r requirements.txt

3️⃣ Add your documents

Place your PDFs in the project root (e.g., medical_book.pdf/).

4️⃣ Run the chatbot
python main.py


You should see:

🤖 RAG Chatbot is ready! Type 'exit' to quit.

You: What are the symptoms of diabetes?
Bot: ...