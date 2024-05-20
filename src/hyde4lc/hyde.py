from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain

from hyde4lc.prompts import default_hyde_prompt_template, defualt_query_prompt_template


def create_hyde_chain(
    llm: BaseLanguageModel,
    retriever: VectorStoreRetriever,
    hyde_prompt_template: BasePromptTemplate = None,
    query_prompt_template: BasePromptTemplate = None,
) -> Runnable[str, BaseMessage]:
    hyde_prompt_template = (
        hyde_prompt_template
        if hyde_prompt_template is not None
        else default_hyde_prompt_template
    )
    query_prompt_template = (
        query_prompt_template
        if query_prompt_template is not None
        else defualt_query_prompt_template
    )

    # List[Document] -> str
    docs_qa_chain = create_stuff_documents_chain(llm, query_prompt_template)

    hyde_chain = (
        RunnablePassthrough().assign(
            context=(
                RunnablePassthrough()
                | (hyde_prompt_template | llm).with_config(run_name="generate_hyp_doc")
                | StrOutputParser()
                | retriever.with_config(run_name="retrieve_doc_with_hyd")
            ).with_config(run_name="retrieve")
        )
        | docs_qa_chain.with_config(run_name="query_with_retrieveds")
    ).with_config(run_name="hyde_qa")

    return hyde_chain
