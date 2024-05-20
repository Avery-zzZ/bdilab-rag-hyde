from langchain_core.prompts import ChatPromptTemplate

default_hyde_prompt_template_str="""Please write a passage that answers the question

###Question
{question}

###Passage
"""
default_hyde_prompt_template = ChatPromptTemplate.from_template(default_hyde_prompt_template_str)

defualt_query_prompt_template_str="""Please answer the following question based only on the provided context:

###Context
```
{context}
```

###Question
{question}

###Answer
"""
defualt_query_prompt_template = ChatPromptTemplate.from_template(defualt_query_prompt_template_str)