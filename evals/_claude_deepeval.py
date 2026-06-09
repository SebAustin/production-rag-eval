"""A Claude-backed DeepEval LLM so G-Eval doesn't require OpenAI.

DeepEval 2.4.1 ships only GPT models; this wraps LangChain's ``ChatAnthropic``
behind ``DeepEvalBaseLLM`` so the financial G-Eval rubric runs on Claude.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepeval.models.base_model import DeepEvalBaseLLM

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic


class ClaudeDeepEval(DeepEvalBaseLLM):
    """DeepEval LLM adapter over ChatAnthropic (temperature 0)."""

    def __init__(self, model_name: str, api_key: str) -> None:
        self._model_name = model_name
        self._api_key = api_key
        super().__init__(model_name)

    def load_model(self) -> ChatAnthropic:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model_name=self._model_name,
            temperature=0,
            timeout=60,
            stop=None,
            api_key=self._api_key,  # type: ignore[arg-type]
        )

    def generate(self, prompt: str, *_args: object, **_kwargs: object) -> str:
        return str(self.load_model().invoke(prompt).content)

    async def a_generate(self, prompt: str, *_args: object, **_kwargs: object) -> str:
        response = await self.load_model().ainvoke(prompt)
        return str(response.content)

    def get_model_name(self) -> str:
        return self._model_name
