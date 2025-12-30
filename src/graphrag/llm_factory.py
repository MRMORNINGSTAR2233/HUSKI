"""LLM factory for creating provider-specific instances."""

import logging
from typing import Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances based on provider."""

    @staticmethod
    def create_chat_llm(config: LLMConfig) -> Any:
        """Create chat LLM instance based on provider.

        Args:
            config: LLM configuration

        Returns:
            Chat LLM instance (ChatOpenAI, ChatGoogleGenerativeAI, or ChatGroq)
        """
        provider = config.provider.lower()

        if provider == "openai":
            logger.info(f"Creating OpenAI chat LLM: {config.openai_model}")
            return ChatOpenAI(
                api_key=config.openai_api_key,
                model=config.openai_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        elif provider == "gemini":
            logger.info(f"Creating Gemini chat LLM: {config.gemini_model}")
            return ChatGoogleGenerativeAI(
                model=config.gemini_model,
                google_api_key=config.gemini_api_key,
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
            )

        elif provider == "groq":
            logger.info(f"Creating Groq chat LLM: {config.groq_model}")
            return ChatGroq(
                api_key=config.groq_api_key,
                model=config.groq_model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: openai, gemini, groq"
            )

    @staticmethod
    def create_embeddings(config: LLMConfig) -> Any:
        """Create embeddings instance based on provider.

        Args:
            config: LLM configuration

        Returns:
            Embeddings instance
        """
        provider = config.provider.lower()

        if provider == "openai":
            logger.info(f"Creating OpenAI embeddings: {config.openai_embedding_model}")
            return OpenAIEmbeddings(
                api_key=config.openai_api_key,
                model=config.openai_embedding_model,
            )

        elif provider == "gemini":
            logger.info(f"Creating Gemini embeddings: {config.gemini_embedding_model}")
            return GoogleGenerativeAIEmbeddings(
                model=config.gemini_embedding_model,
                google_api_key=config.gemini_api_key,
            )

        elif provider == "groq":
            logger.info(
                "Groq doesn't have native embeddings, using OpenAI as fallback"
            )
            # Groq doesn't provide embeddings, fallback to OpenAI
            return OpenAIEmbeddings(
                api_key=config.openai_api_key or config.groq_api_key,
                model=config.groq_embedding_model,
            )

        else:
            raise ValueError(
                f"Unsupported embeddings provider: {provider}. "
                f"Supported providers: openai, gemini, groq"
            )

    @staticmethod
    def get_provider_info(config: LLMConfig) -> dict:
        """Get information about the configured provider.

        Args:
            config: LLM configuration

        Returns:
            Dictionary with provider information
        """
        provider = config.provider.lower()

        provider_info = {
            "openai": {
                "name": "OpenAI",
                "model": config.openai_model,
                "embedding_model": config.openai_embedding_model,
                "context_window": "128K tokens (GPT-4 Turbo)",
                "features": [
                    "Function calling",
                    "JSON mode",
                    "Vision (GPT-4V)",
                    "Large context window",
                ],
            },
            "gemini": {
                "name": "Google Gemini",
                "model": config.gemini_model,
                "embedding_model": config.gemini_embedding_model,
                "context_window": "2M tokens (Gemini 2.5 Pro)",
                "features": [
                    "Multimodal understanding",
                    "Advanced reasoning",
                    "Large context (2M tokens)",
                    "Function calling",
                    "Code execution",
                ],
            },
            "groq": {
                "name": "Groq",
                "model": config.groq_model,
                "embedding_model": f"{config.groq_embedding_model} (via OpenAI)",
                "context_window": "131K tokens (Llama 3.3 70B)",
                "features": [
                    "Ultra-fast inference (~280 tps)",
                    "Open-source models",
                    "Low latency",
                    "Cost-effective",
                ],
            },
        }

        return provider_info.get(
            provider, {"name": "Unknown", "error": f"Unknown provider: {provider}"}
        )
