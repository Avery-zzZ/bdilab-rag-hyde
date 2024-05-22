from langchain_core.prompts import ChatPromptTemplate

default_hyde_prompt_template_str = """The following defines Hypothetical Document Embeddings:
```
Hypothetical Document Embeddings (HyDE) is a technique designed to enhance retrieval performance in LLM-based Retrieve-and-Generate (RAG) tasks. When given a query, HyDE employs a zero-shot approach by prompting a language model that follows instructions to create a "fictional" document. This document, while not real and potentially containing inaccuracies, emulates the content style and relevant information patterns derived from the original query.
```

Please provide answers to the following questions in the form of a hypothetical document (use the same language as the questions)

###Question
{question}

###Answer

"""
default_hyde_prompt_template = ChatPromptTemplate.from_template(
    default_hyde_prompt_template_str
)

defualt_query_prompt_template_str = """Please answer the following questions based solely on the context provided (some information may be irrelevant to the question, please discern this yourself).

###Context
```
{context}
```

###Question
{question}

###Answer
"""
defualt_query_prompt_template = ChatPromptTemplate.from_template(
    defualt_query_prompt_template_str
)
