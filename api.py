from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="RAG Document QA API",
    description="Ask questions about your PDF using RAG",
    version="1.0.0"
)

# Load and process PDF on startup
print("Loading PDF and creating vector database...")
loader = PyPDFLoader("Artificial Intelligence.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer:"""
)

parser = StrOutputParser()

# Request model
class QuestionRequest(BaseModel):
    question: str

# Response model
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]

# Root endpoint
@app.get("/")
def root():
    return {"message": "RAG Document QA API is running!"}

# Ask endpoint
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Retrieve relevant chunks
    relevant_chunks = retriever.invoke(request.question)
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    sources = [doc.page_content[:100] + "..." for doc in relevant_chunks]
    
    # Generate answer
    chain = prompt | llm | parser
    answer = chain.invoke({
        "context": context,
        "question": request.question
    })
    
    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources
    )

print("API Ready!")