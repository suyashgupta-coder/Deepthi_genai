
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. Embedding Model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)

# 2. Connect to ChromaDB
vectorstore = Chroma(
    collection_name="dense_rag",
    embedding_function=embeddings,
    persist_directory="D:/depthi_genai/phase2/part1/vector_store/chroma_db"
)

# 3. Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 4. LLM
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

# 5. Prompt
prompt = ChatPromptTemplate.from_template("""
Answer the question using the given context.

Context: {context}

Question: {question}
""")

print("RAG System is ready!")
print("Type 0 to exit")

# 6. Chatbot
while True:

    question = input("\nYOU: ")

    if question == "0":
        break

    # Retrieve documents
    docs = retriever.invoke(question)

    context = "\n".join(doc.page_content for doc in docs)

    print("\nRetrieved Context:\n", context)

    # Generate Answer
    chain = prompt | model

    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAI:", response.content)
