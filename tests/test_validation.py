"""Tests for agent.validation module"""

import pytest

from agent.validation import (
    ValidationError,
    sanitize_filename,
    sanitize_for_display,
    sanitize_text,
    validate_query,
    validate_state_data,
    validate_topic,
    validate_url,
    validate_urls,
)


def test_validate_topic_success() -> None:
    """Test that validate_topic accepts valid topics"""
    topic = "Machine Learning"
    result = validate_topic(topic)
    assert result == topic


def test_validate_topic_empty() -> None:
    """Test that validate_topic raises error for empty topic"""
    with pytest.raises(ValidationError, match="Topic cannot be empty"):
        validate_topic("")


def test_validate_topic_too_short() -> None:
    """Test that validate_topic raises error for topic that's too short"""
    with pytest.raises(ValidationError, match="at least 3 characters"):
        validate_topic("ab")


def test_validate_topic_too_long() -> None:
    """Test that validate_topic raises error for topic that's too long"""
    long_topic = "A" * 501
    with pytest.raises(ValidationError, match="at most 500 characters"):
        validate_topic(long_topic)


def test_validate_topic_removes_dangerous_patterns() -> None:
    """Test that validate_topic removes dangerous patterns"""
    topic = "Test<script>alert('xss')</script>Topic"
    result = validate_topic(topic)
    assert "<script>" not in result
    assert "alert" not in result


def test_validate_topic_only_whitespace() -> None:
    """Test that validate_topic raises error for only whitespace"""
    with pytest.raises(ValidationError):
        validate_topic("   ")


def test_validate_topic_non_string() -> None:
    """Test that validate_topic converts non-string to string"""
    result = validate_topic(12345)
    assert isinstance(result, str)
    assert result == "12345"


def test_validate_query_success() -> None:
    """Test that validate_query accepts valid queries"""
    query = "quantum computing"
    result = validate_query(query)
    assert result == query


def test_validate_query_empty() -> None:
    """Test that validate_query raises error for empty query"""
    with pytest.raises(ValidationError, match="Query cannot be empty"):
        validate_query("")


def test_validate_query_too_long() -> None:
    """Test that validate_query raises error for query that's too long"""
    long_query = "A" * 201
    with pytest.raises(ValidationError, match="at most 200 characters"):
        validate_query(long_query)


def test_validate_query_only_whitespace() -> None:
    """Test that validate_query raises error for only whitespace"""
    with pytest.raises(ValidationError, match="only whitespace"):
        validate_query("   ")


def test_validate_query_removes_dangerous_patterns() -> None:
    """Test that validate_query removes dangerous patterns"""
    query = "test<script>alert('xss')</script>query"
    result = validate_query(query)
    assert "<script>" not in result


def test_validate_query_non_string() -> None:
    """Test that validate_query converts non-string to string"""
    result = validate_query(12345)
    assert isinstance(result, str)


def test_validate_url_success() -> None:
    """Test that validate_url accepts valid URLs"""
    url = "https://example.com"
    result = validate_url(url)
    assert result == url


def test_validate_url_http() -> None:
    """Test that validate_url accepts http URLs"""
    url = "http://example.com"
    result = validate_url(url)
    assert result.startswith("http://")


def test_validate_url_adds_scheme() -> None:
    """Test that validate_url adds https:// if no scheme"""
    url = "example.com"
    result = validate_url(url)
    assert result.startswith("https://")


def test_validate_url_empty() -> None:
    """Test that validate_url raises error for empty URL"""
    with pytest.raises(ValidationError, match="URL cannot be empty"):
        validate_url("")


def test_validate_url_too_long() -> None:
    """Test that validate_url raises error for URL that's too long"""
    long_url = "https://example.com/" + "a" * 2050
    with pytest.raises(ValidationError, match="at most 2048 characters"):
        validate_url(long_url)


def test_validate_url_dangerous_scheme() -> None:
    """Test that validate_url raises error for dangerous schemes"""
    with pytest.raises(ValidationError):
        validate_url("javascript:alert('xss')")


def test_validate_url_file_scheme() -> None:
    """Test that validate_url raises error for file:// scheme"""
    with pytest.raises(ValidationError):
        validate_url("file:///etc/passwd")


def test_validate_url_custom_allowed_schemes() -> None:
    """Test that validate_url accepts custom allowed schemes"""
    url = "https://example.com"
    result = validate_url(url, allowed_schemes={"https", "http"})
    assert result.startswith("https://")


def test_validate_url_normalizes() -> None:
    """Test that validate_url normalizes URLs"""
    url = "HTTPS://EXAMPLE.COM/path"
    result = validate_url(url)
    assert result == "https://example.com/path"


def test_validate_url_removes_fragment() -> None:
    """Test that validate_url removes fragments"""
    url = "https://example.com#fragment"
    result = validate_url(url)
    assert "#" not in result


def test_validate_urls_success() -> None:
    """Test that validate_urls accepts valid URL list"""
    urls = ["https://example.com", "https://test.com"]
    result = validate_urls(urls)
    assert len(result) == 2
    assert all(url.startswith("https://") for url in result)


def test_validate_urls_empty_list() -> None:
    """Test that validate_urls returns empty list for empty input"""
    result = validate_urls([])
    assert result == []


def test_validate_urls_filters_invalid() -> None:
    """Test that validate_urls filters invalid URLs"""
    urls = ["https://example.com", "javascript:alert('xss')", "https://test.com"]
    result = validate_urls(urls)
    assert len(result) == 2
    assert "javascript:" not in str(result)


def test_validate_urls_max_urls() -> None:
    """Test that validate_urls respects max_urls limit"""
    urls = [f"https://example{i}.com" for i in range(10)]
    result = validate_urls(urls, max_urls=5)
    assert len(result) == 5


def test_validate_urls_not_list() -> None:
    """Test that validate_urls raises error for non-list input"""
    with pytest.raises(ValidationError, match="must be a list"):
        validate_urls("not a list")  # type: ignore[arg-type]


def test_sanitize_text_basic() -> None:
    """Test basic text sanitization"""
    text = "Hello World"
    result = sanitize_text(text)
    assert result == text


def test_sanitize_text_removes_null_bytes() -> None:
    """Test that sanitize_text removes null bytes"""
    text = "Hello\x00World"
    result = sanitize_text(text)
    assert "\x00" not in result


def test_sanitize_text_removes_script_tags() -> None:
    """Test that sanitize_text removes script tags"""
    text = "Hello<script>alert('xss')</script>World"
    result = sanitize_text(text)
    assert "<script>" not in result
    assert "alert" not in result


def test_sanitize_text_max_length() -> None:
    """Test that sanitize_text respects max_length"""
    text = "A" * 100
    result = sanitize_text(text, max_length=50)
    assert len(result) == 50


def test_sanitize_text_normalizes_whitespace() -> None:
    """Test that sanitize_text normalizes whitespace"""
    text = "Hello    World\n\n\nTest"
    result = sanitize_text(text, allow_newlines=True)
    assert "    " not in result
    assert "\n\n\n" not in result


def test_sanitize_text_no_newlines() -> None:
    """Test that sanitize_text removes newlines when allow_newlines=False"""
    text = "Hello\nWorld\nTest"
    result = sanitize_text(text, allow_newlines=False)
    assert "\n" not in result


def test_sanitize_filename_basic() -> None:
    """Test basic filename sanitization"""
    filename = "test_file.txt"
    result = sanitize_filename(filename)
    assert result == filename


def test_sanitize_filename_removes_invalid_chars() -> None:
    """Test that sanitize_filename removes invalid characters"""
    filename = "test<>file|name.txt"
    result = sanitize_filename(filename)
    assert "<" not in result
    assert ">" not in result
    assert "|" not in result


def test_sanitize_filename_empty() -> None:
    """Test that sanitize_filename returns 'untitled' for empty input"""
    result = sanitize_filename("")
    assert result == "untitled"


def test_sanitize_filename_max_length() -> None:
    """Test that sanitize_filename respects max_length"""
    filename = "A" * 300
    result = sanitize_filename(filename, max_length=100)
    assert len(result) <= 100


def test_sanitize_filename_preserves_extension() -> None:
    """Test that sanitize_filename preserves extension when truncating"""
    filename = "A" * 300 + ".txt"
    result = sanitize_filename(filename, max_length=10)
    assert result.endswith(".txt")


def test_sanitize_filename_removes_leading_dots() -> None:
    """Test that sanitize_filename removes leading dots"""
    filename = "...test.txt"
    result = sanitize_filename(filename)
    assert not result.startswith(".")


def test_sanitize_for_display_basic() -> None:
    """Test basic sanitize_for_display"""
    text = "Hello World"
    result = sanitize_for_display(text)
    assert result == text


def test_sanitize_for_display_removes_scripts() -> None:
    """Test that sanitize_for_display removes script tags"""
    text = "Hello<script>alert('xss')</script>World"
    result = sanitize_for_display(text)
    assert "<script>" not in result


def test_sanitize_for_display_truncates() -> None:
    """Test that sanitize_for_display truncates long text"""
    text = "A" * 2000
    result = sanitize_for_display(text, max_length=100)
    assert len(result) <= 103  # 100 + "..."
    assert result.endswith("...")


def test_sanitize_for_display_empty() -> None:
    """Test that sanitize_for_display returns empty string for empty input"""
    result = sanitize_for_display("")
    assert result == ""


def test_sanitize_for_display_non_string() -> None:
    """Test that sanitize_for_display converts non-string to string"""
    result = sanitize_for_display(12345)
    assert isinstance(result, str)


def test_validate_state_data_success() -> None:
    """Test that validate_state_data validates valid state"""
    data = {
        "topic": "Test Topic",
        "search_queries": ["query1", "query2"],
        "search_results": [{"url": "https://example.com"}],
    }
    result = validate_state_data(data)
    assert result["topic"] == "Test Topic"
    assert len(result["search_queries"]) == 2


def test_validate_state_data_not_dict() -> None:
    """Test that validate_state_data raises error for non-dict input"""
    with pytest.raises(ValidationError, match="must be a dictionary"):
        validate_state_data("not a dict")  # type: ignore[arg-type]


def test_validate_state_data_validates_topic() -> None:
    """Test that validate_state_data validates topic"""
    data = {"topic": ""}
    with pytest.raises(ValidationError):
        validate_state_data(data)


def test_validate_state_data_validates_queries() -> None:
    """Test that validate_state_data validates search queries"""
    data = {"search_queries": ["valid query", ""]}
    result = validate_state_data(data)
    # Empty queries should be filtered, whitespace-only should raise
    assert len(result["search_queries"]) >= 1
    assert "valid query" in result["search_queries"]


def test_validate_state_data_validates_urls() -> None:
    """Test that validate_state_data validates URLs in results"""
    data = {
        "search_results": [
            {"url": "https://example.com"},
            {"url": "javascript:alert('xss')"},
        ]
    }
    result = validate_state_data(data)
    # Invalid URLs should be filtered
    assert len(result["search_results"]) <= 1


def test_validate_state_data_copies_other_fields() -> None:
    """Test that validate_state_data copies other fields"""
    data = {
        "topic": "Test",
        "custom_field": "custom_value",
        "another_field": 123,
    }
    result = validate_state_data(data)
    assert result["custom_field"] == "custom_value"
    assert result["another_field"] == 123


def test_validation_error_inheritance() -> None:
    """Test that ValidationError is an Exception"""
    error = ValidationError("test error")
    assert isinstance(error, Exception)
    assert str(error) == "test error"
