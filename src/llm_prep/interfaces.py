from __future__ import annotations

from typing import Any, Protocol, Sequence

from .types import Message, SearchResult, ToolCall


class TokenizerProtocol(Protocol):
    def encode(self, text: str) -> list[int] | list[str]:
        ...

    def decode(self, tokens: Sequence[int] | Sequence[str]) -> str:
        ...


class KVCacheProtocol(Protocol):
    def append(self, key: Any, value: Any) -> None:
        ...

    def get(self) -> tuple[Any, Any]:
        ...

    def reset(self) -> None:
        ...


class SamplerProtocol(Protocol):
    def sample(self, logits: Sequence[float]) -> int:
        ...


class RetrieverProtocol(Protocol):
    def add_document(self, doc_id: str, text: str, embedding: Sequence[float] | None = None) -> None:
        ...

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        ...


class ToolProtocol(Protocol):
    name: str
    description: str

    def call(self, arguments: dict[str, Any]) -> Any:
        ...


class PlannerProtocol(Protocol):
    def plan(self, task: str, available_tools: Sequence[str]) -> list[str]:
        ...


class ChatModelProtocol(Protocol):
    def generate(self, messages: Sequence[Message], tools: Sequence[ToolProtocol] | None = None) -> str | ToolCall:
        ...

