import unittest
from typing import List, Dict
import os
import json

from langchain_core.documents import Document

from src.hyde4lc.hyde import create_hyde_chain
from tests.mocks import MockLLM, MockRetriever
from .util_test_hyde import create_chroma_vs_from_test_set, generate_test_set, evaluate
# replace with your own in code
from playground._langchain.common_steps import embeddings, llm


class TestHyde(unittest.TestCase):

    def test_create_hyde_chain(self):
        llm = MockLLM()
        retriever = MockRetriever(k=2)
        chain = create_hyde_chain(llm, retriever)
        result = chain.invoke({"question": "how can langsmith help with testing?"})
        print(result)

    def test_hyde_evaluation(self):
        test_set = None
        test_set_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "static/test_set.json"))
        with open(test_set_path, 'r', encoding='utf8') as file:
            test_set: Dict = json.load(file)
        
        vs = create_chroma_vs_from_test_set(test_set, embeddings)
        retriever = vs.as_retriever()
        hyde_chain = create_hyde_chain(llm, retriever)

        result_ids = []
        for question in test_set['questions']:
            back_docs = retriever.invoke(question)
            result_id = [doc.metadata["id"] for doc in back_docs]
            result_ids.append(result_id)
        
        print("no hyde evaluation:")
        evaluate(test_set['target_ids'], result_ids)
            
        result_ids = []
        for question in test_set['questions']:
            back_docs: List[Document] = hyde_chain.invoke({"question": question})["context"]
            result_id = [doc.metadata["id"] for doc in back_docs]
            result_ids.append(result_id)
        
        print("hyde evaluation:") 
        evaluate(test_set['target_ids'], result_ids)


if __name__ == "__main__":
    unittest.main()
