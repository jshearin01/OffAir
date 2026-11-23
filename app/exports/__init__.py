"""LLM-ready exports (PRD Phase 5).

Markdown export optimised for pasting into an LLM, plus an optional
local-LLM summariser (Ollama on localhost, or llama-cpp-python) that
is OFF by default and never calls any cloud API.
"""
from app.exports.llm_export import (
    LLM_EXPORT_FILE,
    export_llm_markdown,
    export_llm_markdown_from_session,
)
from app.exports.local_llm import (
    LlamaCppClient,
    LocalLLMConfig,
    LocalLLMError,
    OllamaClient,
    OllamaNotAvailableError,
    summarize_locally,
)

__all__ = [
    "LLM_EXPORT_FILE",
    "export_llm_markdown",
    "export_llm_markdown_from_session",
    "LlamaCppClient",
    "LocalLLMConfig",
    "LocalLLMError",
    "OllamaClient",
    "OllamaNotAvailableError",
    "summarize_locally",
]
