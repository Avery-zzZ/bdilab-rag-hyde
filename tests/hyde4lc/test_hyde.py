import unittest

from src.hyde4lc.hyde import create_hyde_chain
from tests.mocks import MockLLM, MockRetriever

class TestStringMethods(unittest.TestCase):

    def test_create_hyde_chain(self):
        llm = MockLLM()
        retriever = MockRetriever(k = 2)
        chain = create_hyde_chain(llm, retriever)
        result = chain.invoke({"question":"how can langsmith help with testing?"})
        print(result)

if __name__ == '__main__':
    unittest.main()
    