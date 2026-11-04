from dotenv import load_dotenv
load_dotenv()

#way 1 
from langchain_google_genai import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite",temprature=0.6,max_token=100)
# response = model.invoke("tell me about virat kholi")
# print(response.content)

#way2 
# from langchain.chat_models import init_chat_model
# model=init_chat_model("groq:openai/gpt-oss-120b")
# response = model.invoke("why parrot talk?")
# print(response.content)

# #way3
# from langchain.chat_models import init_chat_model
# model=init_chat_model(model="openai/gpt-oss-120b",model_provider="groq")
# response = model.invoke("why parrot talk?")
# print(response.content)

#Huggingface (server)

# from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

# llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b"
# )
# model = ChatHuggingFace(llm=llm)
# response = model.invoke("why parrot talk?")
# print(response.content)

#huggingface locally

from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

model = HuggingFacePipeline(
    pipeline=pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct"
    )
)
response = model.invoke("tell me few good places to visit in USA")
print(response)