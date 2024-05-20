# hyde4lc

## Quick Start

```python
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from hyde4lc import create_hyde_chain

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vs = Chroma.from_documents(documents, embeddings)
retriever = vs.as_retriever()

hyde_chain = create_hyde_chain(llm, retriever)
result = hyde_chain.invoke({"question":"how can langsmith help with testing?"})
print(result)
```