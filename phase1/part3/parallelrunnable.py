from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

#1 prompt template
short_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in simple terms"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "explain {topic} in detailed but it should be in simple terms "
)


#2 model
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")

#parser 
parser = StrOutputParser()

#parallelrunnable

chain=RunnableParallel({
    "short": short_prompt|model|parser,
    "detail":detailed_prompt|model|parser
})

result = chain.invoke({"topic":"Gen Ai"})
print("Short answer : ",result['short'])
print("long ans: ", result['detail']) 