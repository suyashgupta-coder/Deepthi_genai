
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

#1 create a model

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

#2 create a loader

loader = PyPDFLoader("D:/depthi_genai/phase2/part1./sparse_rag/genAI pdf.pdf")
doc = loader.load()

#3 create text splitter 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
chunks = splitter.split_documents(doc)

# Creating a bm25 retriver

retriever = BM25Retriever.from_documents(chunks)
retriever.k=3

question = input("Ask your question:")
results = retriever.invoke(question)
context = "\n \n".join([doc.page_content for doc in results])

prompt = f""" Answer the question using only the context below.
If the answer is not in the context,say you don't know.

Context:
{context}

Question:{question}
"""
res = model.invoke(prompt)
print(res.content[0]['text'])