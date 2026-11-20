from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")
texts=[
    "hello i am suyash gupta",
    "I amm a AI developer ",
    "i am working with techsara"

]

vector = embeddings.embed_documents(texts)
print(vector)

