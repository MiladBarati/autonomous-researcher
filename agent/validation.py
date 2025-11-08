"""
Input Validation and Sanitization Module

Provides functions for validating and sanitizing user inputs to prevent:
- Injection attacks
- XSS vulnerabilities
- Invalid data formats
- Security issues
"""

import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from agent.logger import get_logger

logger = get_logger("validation")


class ValidationError(Exception):
    """Custom exception for validation errors"""

    pass


# Constants for validation
MAX_TOPIC_LENGTH = 500
MIN_TOPIC_LENGTH = 3
MAX_QUERY_LENGTH = 200
MAX_URL_LENGTH = 2048
MAX_FILENAME_LENGTH = 255

# Dangerous patterns to detect
DANGEROUS_PATTERNS = [
    r"<script[^>]*>.*?</script>",  # Script tags
    r"javascript:",  # JavaScript protocol
    r"on\w+\s*=",  # Event handlers (onclick, onerror, etc.)
    r"data:text/html",  # Data URLs with HTML
    r"vbscript:",  # VBScript protocol
    r"file://",  # File protocol
    r"\.\./",  # Path traversal
    r"\.\.\\",  # Path traversal (Windows)
]

# Allowed URL schemes
ALLOWED_URL_SCHEMES = {"http", "https"}

# Invalid filename characters (Windows + Unix)
INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'


def sanitize_text(
    text: str | object, max_length: int | None = None, allow_newlines: bool = True
) -> str:
    """
    Sanitize text input by removing dangerous patterns and normalizing whitespace.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length (None for no limit)
        allow_newlines: Whether to preserve newlines

    Returns:
        Sanitized text
    """
    if not isinstance(text, str):
        text = str(text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Normalize whitespace
    if allow_newlines:
        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Normalize spaces but keep newlines
        text = re.sub(r"[ \t]+", " ", text)
    else:
        # Replace all whitespace with single space
        text = re.sub(r"\s+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    # Apply length limit
    if max_length and len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")

    return text


def validate_topic(topic: str | object) -> str:
    """
    Validate and sanitize research topic.

    Args:
        topic: Research topic to validate

    Returns:
        Sanitized topic

    Raises:
        ValidationError: If topic is invalid
    """
    if not topic:
        raise ValidationError("Topic cannot be empty")

    if not isinstance(topic, str):
        topic = str(topic)

    # Check length
    if len(topic) < MIN_TOPIC_LENGTH:
        raise ValidationError(f"Topic must be at least {MIN_TOPIC_LENGTH} characters long")

    if len(topic) > MAX_TOPIC_LENGTH:
        raise ValidationError(f"Topic must be at most {MAX_TOPIC_LENGTH} characters long")

    # Sanitize
    sanitized = sanitize_text(topic, max_length=MAX_TOPIC_LENGTH, allow_newlines=False)

    # Check if sanitization removed too much (indicates dangerous content)
    if len(sanitized) < MIN_TOPIC_LENGTH:
        raise ValidationError("Topic contains invalid or dangerous content")

    # Check for only whitespace
    if not sanitized.strip():
        raise ValidationError("Topic cannot be only whitespace")

    return sanitized


def validate_query(query: str | object) -> str:
    """
    Validate and sanitize search query.

    Args:
        query: Search query to validate

    Returns:
        Sanitized query

    Raises:
        ValidationError: If query is invalid
    """
    if not query:
        raise ValidationError("Query cannot be empty")

    if not isinstance(query, str):
        query = str(query)

    # Check length
    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(f"Query must be at most {MAX_QUERY_LENGTH} characters long")

    # Sanitize
    sanitized = sanitize_text(query, max_length=MAX_QUERY_LENGTH, allow_newlines=False)

    # Check if result is empty
    if not sanitized.strip():
        raise ValidationError("Query cannot be only whitespace")

    return sanitized


def validate_url(url: str | object, allowed_schemes: set[str] | None = None) -> str:
    """
    Validate and sanitize URL.

    Args:
        url: URL to validate
        allowed_schemes: Set of allowed URL schemes (default: http, https)

    Returns:
        Sanitized and normalized URL

    Raises:
        ValidationError: If URL is invalid
    """
    if not url:
        raise ValidationError("URL cannot be empty")

    if not isinstance(url, str):
        url = str(url)

    # Check length
    if len(url) > MAX_URL_LENGTH:
        raise ValidationError(f"URL must be at most {MAX_URL_LENGTH} characters long")

    # Remove whitespace
    url = url.strip()

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        raise ValidationError(f"Invalid URL format: {e}") from e

    # Check scheme
    if not parsed.scheme:
        # Try adding https:// if no scheme
        url = f"https://{url}"
        try:
            parsed = urlparse(url)
        except Exception as e:
            raise ValidationError(f"Invalid URL format: {e}") from e

    # Validate scheme
    schemes = allowed_schemes or ALLOWED_URL_SCHEMES
    if parsed.scheme.lower() not in schemes:
        raise ValidationError(f"URL scheme must be one of: {', '.join(schemes)}")

    # Check for dangerous patterns in URL
    url_lower = url.lower()
    dangerous_patterns = ["javascript:", "data:", "file:", "vbscript:"]
    for pattern in dangerous_patterns:
        if pattern in url_lower:
            raise ValidationError(f"URL contains dangerous pattern: {pattern}")

    # Reconstruct URL to normalize it
    normalized: str = urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            parsed.params,
            parsed.query,
            "",  # Remove fragment for security
        )
    )

    return normalized


def validate_urls(urls: list[str], max_urls: int | None = None) -> list[str]:
    """
    Validate and sanitize a list of URLs.

    Args:
        urls: List of URLs to validate
        max_urls: Maximum number of URLs to process

    Returns:
        List of validated and sanitized URLs
    """
    if not urls:
        return []

    if not isinstance(urls, list):
        raise ValidationError("URLs must be a list")

    # Limit number of URLs
    if max_urls:
        urls = urls[:max_urls]

    validated_urls: list[str] = []
    for url in urls:
        try:
            validated = validate_url(url)
            validated_urls.append(validated)
        except ValidationError as e:
            logger.warning(f"Skipping invalid URL '{url}': {e}")
            continue

    return validated_urls


def sanitize_filename(filename: str | object, max_length: int | None = None) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Filename to sanitize
        max_length: Maximum filename length (default: MAX_FILENAME_LENGTH)

    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"

    if not isinstance(filename, str):
        filename = str(filename)

    # Remove invalid characters
    sanitized = re.sub(INVALID_FILENAME_CHARS, "_", filename)

    # Remove leading/trailing dots and spaces (Windows restriction)
    sanitized = sanitized.strip(". ")

    # Ensure not empty
    if not sanitized:
        sanitized = "untitled"

    # Apply length limit
    max_len = max_length or MAX_FILENAME_LENGTH
    if len(sanitized) > max_len:
        # Try to preserve extension
        if "." in sanitized:
            name, ext = sanitized.rsplit(".", 1)
            max_name_len = max_len - len(ext) - 1
            sanitized = f"{name[:max_name_len]}.{ext}"
        else:
            sanitized = sanitized[:max_len]

    return sanitized


def validate_state_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and sanitize state data dictionary.

    Args:
        data: State data to validate

    Returns:
        Validated and sanitized state data
    """
    if not isinstance(data, dict):
        raise ValidationError("State data must be a dictionary")

    validated: dict[str, Any] = {}

    # Validate topic if present
    if "topic" in data:
        validated["topic"] = validate_topic(data["topic"])

    # Validate search queries if present
    if "search_queries" in data:
        queries = data["search_queries"]
        if isinstance(queries, list):
            validated["search_queries"] = [validate_query(q) for q in queries if q]
        else:
            validated["search_queries"] = []

    # Validate URLs in various fields
    url_fields = ["search_results", "scraped_content", "arxiv_papers"]
    for field in url_fields:
        if field in data and isinstance(data[field], list):
            validated[field] = []
            for item in data[field]:
                if isinstance(item, dict) and "url" in item:
                    try:
                        item["url"] = validate_url(item["url"])
                        validated[field].append(item)
                    except ValidationError as e:
                        logger.warning(f"Skipping item with invalid URL: {e}")
                        continue
                else:
                    validated[field].append(item)

    # Copy other fields as-is (they'll be validated by their respective handlers)
    for key, value in data.items():
        if key not in validated:
            validated[key] = value

    return validated


def sanitize_for_display(text: str | object, max_length: int = 1000) -> str:
    """
    Sanitize text for safe display in HTML/UI.

    Args:
        text: Text to sanitize
        max_length: Maximum length for display

    Returns:
        Sanitized text safe for display
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove script tags and dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text
