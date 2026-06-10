"""A Claude-backed DeepEval LLM so G-Eval doesn't require OpenAI.

DeepEval 2.4.1 ships only GPT models; this wraps LangChain's ``ChatAnthropic``
behind ``DeepEvalBaseLLM`` so the financial G-Eval rubric runs on Claude.

G-Eval calls ``a_generate(prompt, schema=Steps)`` expecting a parsed pydantic
object back; honor that by coercing the model's JSON reply into the schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from deepeval.models.base_model import DeepEvalBaseLLM

if TYPE_CHECKING:
    from langchain_anthropic import ChatAnthropic
    from pydantic import BaseModel


def _coerce(text: str, schema: type[BaseModel]) -> BaseModel:
    """Parse a (possibly fenced) JSON model reply into ``schema``."""
    cleaned = text.strip()
    if "```" in cleaned:
        for part in cleaned.split("```"):
            candidate = part.strip().removeprefix("json").strip()
            if candidate.startswith("{"):
                cleaned = candidate
                break
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end != -1:
        cleaned = cleaned[start : end + 1]
    return schema.model_validate_json(cleaned)


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

    def generate(self, prompt: str, schema: Any = None) -> Any:  # noqa: ANN401 — DeepEval contract
        text = str(self.load_model().invoke(prompt).content)
        return _coerce(text, schema) if schema is not None else text

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:  # noqa: ANN401
        response = await self.load_model().ainvoke(prompt)
        text = str(response.content)
        return _coerce(text, schema) if schema is not None else text

    def get_model_name(self) -> str:
        return self._model_name
