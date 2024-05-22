from typing import List, Tuple, Any
import os
import re
import json

from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# replace with your own in code
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from playground._langchain.common_steps import llm

static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))

QA_GENERATE_PROMPT_TMPL = ChatPromptTemplate.from_template(
    """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
generate only questions based on the below query in the language of given context.

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
    test_set = {"docs":{}}
    for id, doc in zip(ids, docs):
        questions_str = chain.invoke(
            {"context_str": doc, "num_questions_per_chunk": num_question_per_doc}
        )
        lines = questions_str.strip().splitlines()
        for line in lines:
            line_strip = line.strip()
            if line_strip != "":
                questions.append(re.sub(r"^\d+[\).\s]", "", line).strip())
                target_ids.append([str(id)])
        test_set["docs"][id] = doc
        
    test_set["questions"] = questions
    test_set["target_ids"] = target_ids
    
    with open(os.path.join(static_path, "test_set.json"), 'w', encoding='utf-8') as json_file:
        json.dump(test_set, json_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    docs: List[str] = []
    for root, dirs, files in os.walk(
        os.path.join(static_path, "docs")
    ):
        for file in files:
            with open(os.path.join(root, file), 'r', encoding='utf8') as file:
                file_content = file.read()
                paragraphs = file_content.split('\n\n')
                for p in paragraphs:
                    docs.append(p)

    ids = [i for i in range(len(docs))]
    generate_test_set(docs, ids, llm)
    
    

