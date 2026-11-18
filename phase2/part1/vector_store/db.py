
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from dotenv import load_dotenv
load_dotenv()




# 1 documenent loader
loader = PyPDFLoader("D:/depthi_genai/phase2/part1/document loaders/genAI pdf.pdf")
docs = loader.load()

#2 Text splitter 
splitter =RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
#3 embbeding model 
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

print("total chunks:",len(chunks))

#4 store chunks in Chroma DB

vectorstore = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory="D:/depthi_genai/phase2/part1/vector_store/chroma_db",
    collection_name="dense_rag"
)
print("vector store created ")
print("Total stored chunks:", vectorstore._collection.count())