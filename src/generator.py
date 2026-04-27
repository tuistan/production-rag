"""Generation Module: context assembly + LLM call.
Implemented on Day 32.
"""


class Generator:
    """Assembles context and generates answers."""

    def __init__(self, config):
        self.config = config

    def build_prompt(self, query: str, contexts: list[dict]) -> str:
        """Assemble system prompt + retrieved contexts + question.
        Includes anti-hallucination instruction.
        """
        pass

    def generate(self, prompt: str) -> str:
        """Call LLM (flan-t5 / OpenAI / Anthropic) based on config.
        Day 32: implement with provider switching
        """
        pass

    def answer(self, query: str, contexts: list[dict]) -> dict:
        """End-to-end: build_prompt -> generate.
        Returns {"answer": str, "prompt": str, "model": str}
        """
        pass