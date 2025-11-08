"""Tests for main.py module"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from agent.validation import ValidationError


def test_main_without_topic() -> None:
    """Test main() without topic argument"""
    with (
        patch("main.logger") as mock_logger,
        patch("sys.exit") as mock_exit,
        patch("main.argparse.ArgumentParser") as MockParser,
    ):
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.topic = None
        mock_args.web = False
        mock_parser.parse_args.return_value = mock_args
        MockParser.return_value = mock_parser

        from main import main

        main()
        # May be called multiple times due to error handling
        assert mock_exit.called


def test_main_with_invalid_topic() -> None:
    """Test main() with invalid topic"""
    import subprocess

    with (
        patch("main.logger") as mock_logger,
        patch("sys.exit") as mock_exit,
        patch("main.argparse.ArgumentParser") as MockParser,
        patch("subprocess.run") as mock_run,
    ):
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.topic = "ab"  # Too short
        mock_args.web = False
        mock_parser.parse_args.return_value = mock_args
        MockParser.return_value = mock_parser

        from main import main

        main()
        mock_exit.assert_called_once_with(1)


def test_main_with_web_flag() -> None:
    """Test main() with --web flag"""
    with (
        patch("main.logger"),
        patch("main.subprocess.run") as mock_run,
        patch("main.argparse.ArgumentParser") as MockParser,
    ):
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.topic = None
        mock_args.web = True
        mock_parser.parse_args.return_value = mock_args

        from main import main

        main()
        mock_run.assert_called_once_with(["streamlit", "run", "app.py"])


def test_main_with_valid_topic() -> None:
    """Test main() with valid topic"""
    with (
        patch("main.logger"),
        patch("main.Config.validate", return_value=True),
        patch("main.create_research_graph") as mock_create,
        patch("main.set_logging_context"),
        patch("builtins.open", create=True) as mock_open,
        patch("main.validate_topic", return_value="Test Topic"),
    ):
        mock_agent = MagicMock()
        mock_agent.research.return_value = {
            "status": "completed",
            "synthesis": "Test synthesis",
        }
        mock_create.return_value = mock_agent

        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.topic = "Test Topic"
        mock_args.web = False

        with patch("main.argparse.ArgumentParser") as MockParser:
            MockParser.return_value.parse_args.return_value = mock_args

            from main import main

            main()
            mock_agent.research.assert_called_once_with("Test Topic")


def test_main_handles_exception() -> None:
    """Test main() handles exceptions gracefully"""
    import subprocess

    with (
        patch("main.logger") as mock_logger,
        patch("sys.exit") as mock_exit,
        patch("main.argparse.ArgumentParser") as MockParser,
        patch("subprocess.run") as mock_run,
    ):
        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.topic = "Test Topic"
        mock_args.web = False
        mock_parser.parse_args.return_value = mock_args
        MockParser.return_value = mock_parser

        with patch("main.Config.validate", side_effect=Exception("Test error")):
            from main import main

            main()
            mock_exit.assert_called_once_with(1)

