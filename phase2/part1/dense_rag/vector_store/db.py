
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# 1. Document Loader
loader = PyPDFLoader(
    "D:/depthi_genai/phase2/part1/document loaders/genAI pdf.pdf"
)

docs = loader.load()

print("Total pages loaded:", len(docs))

# 2. Text Splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Total chunks created:", len(chunks))

if len(chunks) == 0:
    raise ValueError("No chunks generated. Check your PDF file.")

# 3. Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# Test Embedding
test_embedding = embeddings.embed_query("What is Generative AI?")
print("Embedding model working!")

# 4. Store in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="D:/depthi_genai/phase2/part1/vector_store/chroma_db",
    collection_name="dense_rag"
)

print("Vector store created successfully!")

print("Total stored chunks:", vectorstore._collection.count())
