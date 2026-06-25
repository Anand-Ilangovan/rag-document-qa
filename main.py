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

# Step 1 — Load the PDF
print("Loading PDF...")
loader = PyPDFLoader("Artificial Intelligence.pdf")
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# Step 2 — Split into chunks
print("Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3 — Create embeddings and store in ChromaDB
print("Creating vector database...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(chunks, embeddings)
print("Vector database ready!")

# Step 4 — Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Step 5 — Create LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# Step 6 — Create prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context: {context}

Question: {question}

Answer:"""
)

parser = StrOutputParser()

# Step 7 — QA function
def ask(question):
    # Retrieve relevant chunks
    relevant_chunks = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    
    # Generate answer
    chain = prompt | llm | parser
    answer = chain.invoke({
        "context": context,
        "question": question
    })
    return answer

# Step 8 — Chat loop
if __name__ == "__main__":
    print("\n=== RAG Document QA ===")
    print("Ask anything about your PDF. Type 'quit' to exit.\n")
    
    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = ask(question)
        print(f"\nAI: {answer}\n")