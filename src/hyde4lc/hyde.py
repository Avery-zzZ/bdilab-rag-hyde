from typing import List, Dict, Any

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain

from hyde4lc.prompts import default_hyde_prompt_template

def print_hyd(x):
    print(x)
    return x

def create_hyde_chain(
    llm: BaseLanguageModel,
    retriever: VectorStoreRetriever,
    hyde_prompt_template: BasePromptTemplate = None,
) -> Runnable[Dict[str, Any], List[Dict]]:
    hyde_prompt_template = (
        hyde_prompt_template
        if hyde_prompt_template is not None
        else default_hyde_prompt_template
    )

    hyde_chain = (
        RunnablePassthrough()
        .assign(
            context=(
                RunnablePassthrough()
                | (hyde_prompt_template | llm).with_config(run_name="generate_hyp_doc")
                | StrOutputParser() | print_hyd
                | retriever.with_config(run_name="retrieve_doc_with_hyd")
            ).with_config(run_name="retrieve")
        ).with_config(run_name="hyde_chain")
    )

    return hyde_chain
