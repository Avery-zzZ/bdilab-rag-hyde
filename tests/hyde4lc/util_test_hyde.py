from typing import List, Tuple, Any, Dict
import re

from langchain_community.vectorstores.qdrant import Qdrant
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


def create_qdrant_vs_from_files(paths: List[str]) -> Qdrant: ...


def create_chroma_vs_from_test_set(test_set: Dict, embeddings: Embeddings) -> Chroma:
    docs: List[Document] = []
    for doc_id, doc_str in test_set["docs"].items():
        doc = Document(doc_str)
        doc.metadata["id"] = doc_id
        docs.append(doc)

    return Chroma.from_documents(docs, embeddings)

QA_GENERATE_PROMPT_TMPL = ChatPromptTemplate.from_template(
    """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""
)


def generate_test_set(
    docs: List[str],
    ids: List[str],
    llm: BaseLanguageModel,
    num_question_per_doc: int = 2,
) -> Tuple[List[str], List[List[Any]]]:
    chain = QA_GENERATE_PROMPT_TMPL | llm | StrOutputParser()
    questions = []
    target_ids = []
    for doc, id in zip(docs, ids):
        questions_str = chain.invoke(
            {"context_str": doc, "num_questions_per_chunk": num_question_per_doc}
        )
        questions_raw = questions_str.strip().split("\n")
        for question in questions_raw:
            questions.append(re.sub(r"^\d+[\).\s]", "", question).strip())
            target_ids.append([id])
    return questions, target_ids


def evaluate(target_ids: List[List[Any]], result_ids: List[List[Any]]) -> None:
    hits = []
    mrr_scores = []

    for target_id, result_id in zip(target_ids, result_ids):
        hit = any(i in target_id for i in result_id)
        hits.append(int(hit))

        rr = 0
        for rank, id in enumerate(result_id, start=1):
            if id in target_id:
                rr = 1 / rank
                break
        mrr_scores.append(rr)

    hit_rate = sum(hits) / len(hits)
    mrr = sum(mrr_scores) / len(mrr_scores)

    print({"hit_rate": hit_rate, "mrr": mrr})
