from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("D:/depthi_genai/phase2/part1/document loaders/genAI pdf.pdf")
docs = loader.load()
print(docs[5].page_content)
print(len(docs))
