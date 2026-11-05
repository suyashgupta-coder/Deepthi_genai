from dotenv import load_dotenv
load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
model = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

print("__________Welcome to my chatbot press 0 to exit__________")
print("1. Funny")
print("2. Angry")
print("3. sad")
print("4 Romantic")
personalities ={
    "1":"you are a funny chatbot who responds with humor",
    "2":"you are a angry chatbot who responds with irritated tone",
    "3": "you are a sad chatbot who responds in a melancholic tone",
    "4":"you are a romantic chatbot who reponds in a romantic tone"

}
choice = input("\n Choose personality(1-4):")
if choice not in personalities:
    print("invalid choice")
    exit()

messages =[
    SystemMessage(content=personalities[choice])
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
    # print(messages)
