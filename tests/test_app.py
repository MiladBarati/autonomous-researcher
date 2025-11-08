"""Tests for app.py module"""

import sys
from unittest.mock import MagicMock, patch

from agent.validation import ValidationError

# Mock streamlit before importing app
mock_streamlit = MagicMock()
sys.modules["streamlit"] = mock_streamlit
mock_st = MagicMock()
mock_streamlit.st = mock_st


def test_initialize_session_state() -> None:
    """Test initialize_session_state"""
    mock_session_state = MagicMock()
    mock_session_state.research_results = None
    mock_session_state.research_history = []
    mock_session_state.agent = None

    with patch("app.st.session_state", mock_session_state):
        from app import initialize_session_state

        initialize_session_state()
        # Should not raise


def test_display_header() -> None:
    """Test display_header"""
    with (
        patch("app.st.title") as mock_title,
        patch("app.st.markdown") as mock_markdown,
        patch("app.st.divider") as mock_divider,
    ):
        from app import display_header

        display_header()
        mock_title.assert_called_once()
        mock_markdown.assert_called_once()
        mock_divider.assert_called_once()


def test_display_sidebar() -> None:
    """Test display_sidebar"""
    with (
        patch("app.st.sidebar"),
        patch("app.st.session_state", {"research_history": []}),
        patch("app.Config") as MockConfig,
    ):
        MockConfig.MODEL_NAME = "test-model"
        MockConfig.MODEL_TEMPERATURE = 0.7
        MockConfig.MAX_TOKENS = 1000
        MockConfig.CHUNK_SIZE = 1000
        MockConfig.EMBEDDING_MODEL = "test-embedding"
        MockConfig.TOP_K_RESULTS = 5
        MockConfig.MAX_SEARCH_RESULTS = 5
        MockConfig.MAX_ARXIV_RESULTS = 3
        MockConfig.MAX_SCRAPE_URLS = 5

        from app import display_sidebar

        display_sidebar()
        # Should not raise


def test_display_research_input() -> None:
    """Test display_research_input"""
    with (
        patch("app.st.header"),
        patch("app.st.columns") as mock_columns,
        patch("app.st.text_input") as mock_text_input,
        patch("app.st.button") as mock_button,
    ):
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_columns.return_value = [mock_col1, mock_col2]
        mock_text_input.return_value = "Test Topic"
        mock_button.return_value = False

        from app import display_research_input

        topic, start_button = display_research_input()
        assert topic == "Test Topic"
        assert start_button is False


def test_display_progress_tracking() -> None:
    """Test display_progress_tracking"""
    from agent.state import ResearchState

    state: ResearchState = {
        "topic": "Test Topic",
        "request_id": "test-id",
        "status": "planned",
        "step_count": 1,
        "research_plan": None,
        "search_queries": [],
        "search_results": [],
        "scraped_content": [],
        "arxiv_papers": [],
        "pdf_content": [],
        "all_documents": [],
        "vector_store_id": None,
        "retrieved_chunks": [],
        "synthesis": None,
        "messages": [],
    }

    with (
        patch("app.st.subheader"),
        patch("app.st.progress") as mock_progress,
        patch("app.st.columns") as mock_columns,
        patch("app.st.metric"),
        patch("app.st.markdown"),
    ):
        mock_columns.return_value = [MagicMock(), MagicMock(), MagicMock()]

        from app import display_progress_tracking

        display_progress_tracking(state)
        mock_progress.assert_called_once()


def test_display_research_results() -> None:
    """Test display_research_results"""
    from agent.state import ResearchState

    state: ResearchState = {
        "synthesis": "Test synthesis",
        "topic": "Test Topic",
        "request_id": "test-id",
        "research_plan": "Test plan",
        "search_queries": [],
        "search_results": [{"title": "Result", "url": "http://test.com", "score": 0.9}],
        "scraped_content": [{"title": "Content", "url": "http://test.com", "word_count": 100}],
        "arxiv_papers": [{"title": "Paper", "authors": ["Author"], "published": "2024-01-01"}],
        "pdf_content": [],
        "all_documents": [{"source": "web"}],
        "vector_store_id": None,
        "retrieved_chunks": [{"text": "Chunk", "metadata": {"title": "Source"}}],
        "messages": [],
        "step_count": 1,
        "status": "completed",
    }

    with (
        patch("app.st.header") as mock_header,
        patch("app.st.subheader"),
        patch("app.st.markdown"),
        patch("app.st.download_button"),
        patch("app.st.divider"),
        patch("app.st.columns") as mock_columns,
        patch("app.st.expander") as mock_expander,
        patch("app.datetime") as mock_datetime,
    ):
        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_expander.return_value.__enter__.return_value = MagicMock()

        from app import display_research_results

        display_research_results(state)
        mock_header.assert_called_once()


def test_run_research_success() -> None:
    """Test run_research with valid topic"""
    with (
        patch("app.st.session_state", {"agent": None}),
        patch("app.st.spinner") as mock_spinner,
        patch("app.create_research_graph") as mock_create,
        patch("app.Config.validate", return_value=True),
        patch("app.validate_topic", return_value="Test Topic"),
    ):
        mock_agent = MagicMock()
        mock_agent.research.return_value = {
            "status": "completed",
            "synthesis": "Test synthesis",
        }
        mock_create.return_value = mock_agent

        mock_spinner.return_value.__enter__.return_value = None
        mock_spinner.return_value.__exit__.return_value = None

        from app import run_research

        result = run_research("Test Topic")
        assert result is not None
        assert result["status"] == "completed"


def test_run_research_invalid_topic() -> None:
    """Test run_research with invalid topic"""
    with (
        patch("app.st.error") as mock_error,
        patch("app.validate_topic", side_effect=ValidationError("Invalid topic")),
    ):
        from app import run_research

        result = run_research("ab")
        assert result is None
        mock_error.assert_called_once()


def test_run_research_exception() -> None:
    """Test run_research handles exceptions"""
    with (
        patch("app.st.error") as mock_error,
        patch("app.st.exception"),
        patch("app.validate_topic", side_effect=Exception("Test error")),
    ):
        from app import run_research

        result = run_research("Test Topic")
        assert result is None
        mock_error.assert_called_once()


def test_main() -> None:
    """Test main function"""
    with (
        patch("app.initialize_session_state") as mock_init,
        patch("app.display_header") as mock_header,
        patch("app.display_sidebar") as mock_sidebar,
        patch("app.display_research_input", return_value=("", False)) as mock_input,
        patch("app.st.session_state", {"research_results": None}),
        patch("app.st.divider"),
        patch("app.st.markdown"),
    ):
        from app import main

        main()
        mock_init.assert_called_once()
        mock_header.assert_called_once()
        mock_sidebar.assert_called_once()
        mock_input.assert_called_once()
