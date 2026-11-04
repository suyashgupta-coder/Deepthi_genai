from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

print("__________Welcome to my chatbot press 0 to exit__________")
messages =[
    SystemMessage(content="you are a romantic chatbot")
]
while True:
    prompt = input("YOU:")
    messages.append(HumanMessage(content=prompt))
    if prompt== "0":
        print("\n Thank you for using my chatbot \n")
        break
    response = model.invoke(messages)
    messages.append(AIMessage(content=response.content[0]["text"]))
    print(response.content[0]["text"])
    print(messages)
