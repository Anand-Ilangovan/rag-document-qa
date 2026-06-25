# RAG Document QA

Ask questions about any PDF using Retrieval Augmented Generation (RAG).
Upload a document and get accurate answers powered by Llama 3 — no hallucinations,
answers come strictly from your document.

## How it works
1. PDF is loaded and split into small chunks
2. Each chunk is converted into a vector embedding using HuggingFace
3. Embeddings are stored in ChromaDB (vector database)
4. When you ask a question, the most relevant chunks are retrieved
5. Llama 3 answers using only those chunks as context

## Architecture
PDF → PyPDFLoader → Text Splitter → HuggingFace Embeddings → ChromaDB → Retriever → Llama 3 → Answer

## Tech Stack
- Python
- LangChain
- ChromaDB (vector database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq API (Llama 3.3 70B)
- python-dotenv

## How to Run
1. Clone the repo
2. Create a `.env` file with your `GROQ_API_KEY`
3. Add your PDF file to the project folder
4. Install dependencies:
   `pip install langchain langchain-groq langchain-community langchain-text-splitters chromadb pypdf sentence-transformers python-dotenv`
5. Update the filename in `main.py` to match your PDF
6. Run: `python main.py`
7. Ask questions about your document!

## Example
You: How is AI used in medical imaging?

AI: AI-powered tools can detect cancer, tumors, and other abnormalities 
in X-rays, MRIs, and CT scans with accuracy comparable to experienced 
radiologists. Companies like Google and IBM have developed models that 
outperform humans in detecting certain cancers.
