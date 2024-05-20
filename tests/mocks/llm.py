from typing import Any

from langchain_core.language_models.llms import LLM

class MockLLM(LLM):
    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        return f"llm response to query ({prompt})"
    
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "llm_test_mock"