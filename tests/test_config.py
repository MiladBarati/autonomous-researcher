"""Tests for config module"""

import os
from unittest.mock import MagicMock, patch

import pytest

from config import Config, get_llm


def test_config_has_required_attributes() -> None:
    """Test that Config has all required configuration attributes"""
    assert hasattr(Config, "GROQ_API_KEY")
    assert hasattr(Config, "TAVILY_API_KEY")
    assert hasattr(Config, "MODEL_NAME")
    assert hasattr(Config, "MODEL_TEMPERATURE")
    assert hasattr(Config, "MAX_TOKENS")
    assert hasattr(Config, "CHUNK_SIZE")
    assert hasattr(Config, "CHUNK_OVERLAP")
    assert hasattr(Config, "EMBEDDING_MODEL")
    assert hasattr(Config, "TOP_K_RESULTS")
    assert hasattr(Config, "CHROMA_PERSIST_DIR")
    assert hasattr(Config, "COLLECTION_NAME")
    assert hasattr(Config, "MAX_SEARCH_RESULTS")
    assert hasattr(Config, "MAX_ARXIV_RESULTS")
    assert hasattr(Config, "MAX_SCRAPE_URLS")


def test_config_validate_missing_keys() -> None:
    """Test that Config.validate raises error for missing keys"""
    with (
        patch.dict(os.environ, {}, clear=True),
        patch.object(Config, "GROQ_API_KEY", None),
        patch.object(Config, "TAVILY_API_KEY", None),
        pytest.raises(ValueError, match="Missing required API keys"),
    ):
        Config.validate()


def test_config_validate_missing_groq_key() -> None:
    """Test that Config.validate raises error for missing GROQ key"""
    from pydantic import SecretStr

    with (
        patch.object(Config, "GROQ_API_KEY", None),
        patch.object(Config, "TAVILY_API_KEY", SecretStr("test-tavily-key")),
        pytest.raises(ValueError, match="Missing required API keys"),
    ):
        Config.validate()


def test_config_validate_missing_tavily_key() -> None:
    """Test that Config.validate raises error for missing Tavily key"""
    from pydantic import SecretStr

    with (
        patch.object(Config, "GROQ_API_KEY", SecretStr("test-groq-key")),
        patch.object(Config, "TAVILY_API_KEY", None),
        pytest.raises(ValueError, match="Missing required API keys"),
    ):
        Config.validate()


def test_config_validate_success() -> None:
    """Test that Config.validate returns True when keys are present"""
    with (
        patch.object(Config, "GROQ_API_KEY", "test-groq-key"),
        patch.object(Config, "TAVILY_API_KEY", "test-tavily-key"),
    ):
        result = Config.validate()
        assert result is True


def test_get_llm_returns_chatgroq() -> None:
    """Test that get_llm returns a ChatGroq instance"""
    with (
        patch.object(Config, "validate", return_value=True),
        patch("config.ChatGroq") as MockChatGroq,
        patch.object(Config, "GROQ_API_KEY", "test-key"),
        patch.object(Config, "MODEL_NAME", "test-model"),
        patch.object(Config, "MODEL_TEMPERATURE", 0.7),
        patch.object(Config, "MAX_TOKENS", 1000),
    ):
        mock_llm = MagicMock()
        MockChatGroq.return_value = mock_llm

        llm = get_llm()
        assert llm is mock_llm
        MockChatGroq.assert_called_once()


def test_get_llm_with_custom_temperature() -> None:
    """Test that get_llm accepts custom temperature"""
    with (
        patch.object(Config, "validate", return_value=True),
        patch("config.ChatGroq") as MockChatGroq,
        patch.object(Config, "GROQ_API_KEY", "test-key"),
        patch.object(Config, "MODEL_NAME", "test-model"),
        patch.object(Config, "MODEL_TEMPERATURE", 0.7),
        patch.object(Config, "MAX_TOKENS", 1000),
    ):
        mock_llm = MagicMock()
        MockChatGroq.return_value = mock_llm

        get_llm(temperature=0.5)
        # Verify temperature was passed
        call_kwargs = MockChatGroq.call_args[1]
        assert call_kwargs["temperature"] == 0.5


def test_get_llm_with_custom_max_tokens() -> None:
    """Test that get_llm accepts custom max_tokens"""
    with (
        patch.object(Config, "validate", return_value=True),
        patch("config.ChatGroq") as MockChatGroq,
        patch.object(Config, "GROQ_API_KEY", "test-key"),
        patch.object(Config, "MODEL_NAME", "test-model"),
        patch.object(Config, "MODEL_TEMPERATURE", 0.7),
        patch.object(Config, "MAX_TOKENS", 1000),
    ):
        mock_llm = MagicMock()
        MockChatGroq.return_value = mock_llm

        get_llm(max_tokens=2000)
        # Verify max_tokens was passed
        call_kwargs = MockChatGroq.call_args[1]
        assert call_kwargs["max_tokens"] == 2000


def test_get_llm_uses_defaults() -> None:
    """Test that get_llm uses default temperature and max_tokens"""
    with (
        patch.object(Config, "validate", return_value=True),
        patch("config.ChatGroq") as MockChatGroq,
        patch.object(Config, "GROQ_API_KEY", "test-key"),
        patch.object(Config, "MODEL_NAME", "test-model"),
        patch.object(Config, "MODEL_TEMPERATURE", 0.7),
        patch.object(Config, "MAX_TOKENS", 1000),
    ):
        mock_llm = MagicMock()
        MockChatGroq.return_value = mock_llm

        get_llm()
        # Verify defaults were used
        call_kwargs = MockChatGroq.call_args[1]
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 1000
