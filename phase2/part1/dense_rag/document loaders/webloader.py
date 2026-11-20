from langchain_community.document_loaders import WebBaseLoader

url = "ttps://www.apple.com/in/macbook-neo/"
loader = WebBaseLoader(url)
docs = loader.load()
print(docs[0])
