from langchain_community.document_loaders import TextLoader

loader = TextLoader("D:/depthi_genai/phase2/part1/document loaders/notes.txt")
docs = loader.load()
print(docs[0])
