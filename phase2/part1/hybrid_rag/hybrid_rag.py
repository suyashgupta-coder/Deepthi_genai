
from dotenv import load_dotenv
load_dotenv()

# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI


# 1. Load PDF
loader = PyPDFLoader(
    "D:/depthi_genai/phase2/part1/hybrid_rag/genAI pdf.pdf"
)

docs = loader.load()

print("Total pages:", len(docs))


# 2. Split Documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

print("Total chunks:", len(chunks))


# 3. Create Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# 4. Create Chroma Vector Database
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="hybrid_rag"
)


# 5. Dense Retriever
dense_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# 6. Sparse Retriever
sparse_retriever = BM25Retriever.from_documents(chunks)

sparse_retriever.k = 5


# 7. Combine Dense + Sparse
retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.5, 0.5]
)


# 8. Load Gemini Model
model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash"
)


# 9. Chat Loop
print("\nHybrid RAG System is Ready!")
print("Type 0 to exit\n")

while True:

    question = input("YOU: ")

    if question == "0":
        print("Goodbye!")
        break

    # Retrieve relevant documents
    results = retriever.invoke(question)

    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    # Display retrieved context
    print("\nRetrieved Context:\n")
    print(context)

    # Create Prompt
    prompt = f"""
    You are a helpful AI assistant.

    Answer the question using ONLY the provided context.

    If the answer is not available in the context,
    clearly say that you don't know based on the PDF.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate Answer
    response = model.invoke(prompt)

    print("\nAI:", response.text)
    print("\n" + "-" * 60 + "\n")
