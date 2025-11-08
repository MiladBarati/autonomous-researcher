"""
Research Tools for Autonomous Agent

Implements LangChain-compatible tools for:
- Web search (Tavily)
- Web scraping (BeautifulSoup)
- ArXiv paper search
- PDF processing
- Vector store operations
"""

import io
import re
from typing import Any

import arxiv
import PyPDF2
import requests
from bs4 import BeautifulSoup
from langchain_core.tools import Tool
from tavily import TavilyClient

from agent.logger import get_logger
from agent.validation import ValidationError, validate_query, validate_url, validate_urls
from config import Config

logger = get_logger("tools")


class TavilySearchTool:
    """Web search using Tavily API"""

    def __init__(self) -> None:
        api_key = Config.TAVILY_API_KEY.get_secret_value() if Config.TAVILY_API_KEY else None
        self.client: TavilyClient = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Execute web search using Tavily.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of search results with title, url, content, and score
        """
        try:
            # Validate and sanitize query
            try:
                query = validate_query(query)
            except ValidationError as e:
                logger.error(f"Invalid search query: {e}")
                return []

            max_results_value: int = max_results or Config.MAX_SEARCH_RESULTS
            response: dict[str, Any] = self.client.search(
                query=query,
                max_results=max_results_value,
                search_depth="advanced",
                include_answer=True,
            )

            results: list[dict[str, Any]] = []
            for result in response.get("results", []):
                results.append(
                    {
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", ""),
                        "score": result.get("score", 0.0),
                        "source": "tavily",
                    }
                )

            return results

        except requests.RequestException as e:
            logger.error(f"Tavily API network error: {e}", exc_info=True)
            return []
        except KeyError as e:
            logger.error(f"Tavily API response format error: {e}", exc_info=True)
            return []
        except ValueError as e:
            logger.error(f"Tavily API validation error: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Tavily search unexpected error: {e}", exc_info=True)
            return []

    def as_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="web_search",
            description="Search the web for information on a given topic. Use this to find current information and relevant sources.",
            func=lambda query: str(self.search(query)),
        )


class WebScraperTool:
    """Web content extraction using BeautifulSoup"""

    def __init__(self) -> None:
        self.headers: dict[str, str] = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

    def scrape(self, url: str) -> dict[str, Any]:
        """
        Extract main content from a webpage.

        Args:
            url: URL to scrape

        Returns:
            Dictionary with url, title, content, and metadata
        """
        try:
            # Validate and sanitize URL
            try:
                url = validate_url(url)
            except ValidationError as e:
                logger.error(f"Invalid URL: {e}")
                return {
                    "url": url,
                    "title": "",
                    "content": "",
                    "word_count": 0,
                    "source": "web_scrape",
                    "success": False,
                    "error": str(e),
                }

            response: requests.Response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup: BeautifulSoup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get title
            title = soup.find("title")
            title_text: str = title.get_text().strip() if title else ""

            # Get main content
            # Try to find main content area
            main_content = soup.find("main") or soup.find("article") or soup.find("body")

            content: str
            if main_content and hasattr(main_content, "find_all"):
                # Extract text from paragraphs
                paragraphs = main_content.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
                content = "\n\n".join(
                    [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
                )
            else:
                content = soup.get_text()

            # Clean up whitespace
            content = re.sub(r"\n\s*\n", "\n\n", content)
            content = re.sub(r" +", " ", content)

            return {
                "url": url,
                "title": title_text,
                "content": content[:10000],  # Limit to 10k chars
                "word_count": len(content.split()),
                "source": "web_scrape",
                "success": True,
            }

        except requests.Timeout as e:
            logger.error(f"Request timeout for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "source": "web_scrape",
                "success": False,
                "error": f"Timeout: {str(e)}",
            }
        except requests.ConnectionError as e:
            logger.error(f"Connection error for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "source": "web_scrape",
                "success": False,
                "error": f"Connection error: {str(e)}",
            }
        except requests.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "source": "web_scrape",
                "success": False,
                "error": f"HTTP {e.response.status_code if hasattr(e, 'response') else 'error'}: {str(e)}",
            }
        except requests.RequestException as e:
            logger.error(f"Network error for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "source": "web_scrape",
                "success": False,
                "error": str(e),
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid input for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "source": "web_scrape",
                "success": False,
                "error": str(e),
            }
        except AttributeError as e:
            logger.warning(f"HTML parsing error for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "source": "web_scrape",
                "success": False,
                "error": f"Parsing error: {str(e)}",
            }
        except Exception as e:
            logger.warning(f"Unexpected scraping error for {url}: {e}")
            return {
                "url": url,
                "title": "",
                "content": "",
                "word_count": 0,
                "source": "web_scrape",
                "success": False,
                "error": str(e),
            }

    def scrape_multiple(self, urls: list[str], max_urls: int | None = None) -> list[dict[str, Any]]:
        """
        Scrape multiple URLs.

        Args:
            urls: List of URLs to scrape
            max_urls: Maximum number of URLs to process

        Returns:
            List of scraped content dictionaries
        """
        # Validate and sanitize URLs
        try:
            urls = validate_urls(urls, max_urls=max_urls or Config.MAX_SCRAPE_URLS)
        except ValidationError as e:
            logger.error(f"Invalid URLs: {e}")
            return []

        max_urls_value: int = max_urls or Config.MAX_SCRAPE_URLS
        results: list[dict[str, Any]] = []

        for url in urls[:max_urls_value]:
            result: dict[str, Any] = self.scrape(url)
            if result["success"] and result["content"]:
                results.append(result)

        return results

    def as_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="web_scraper",
            description="Extract full content from a webpage URL. Use this to get detailed information from specific sources.",
            func=lambda url: str(self.scrape(url)),
        )


class ArxivSearchTool:
    """Academic paper search using ArXiv API"""

    def search(self, query: str, max_results: int | None = None) -> list[dict[str, Any]]:
        """
        Search ArXiv for academic papers.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of paper metadata
        """
        try:
            # Validate and sanitize query
            try:
                query = validate_query(query)
            except ValidationError as e:
                logger.error(f"Invalid ArXiv search query: {e}")
                return []

            max_results_value: int = max_results or Config.MAX_ARXIV_RESULTS

            search: arxiv.Search = arxiv.Search(
                query=query, max_results=max_results_value, sort_by=arxiv.SortCriterion.Relevance
            )

            results: list[dict[str, Any]] = []
            for paper in search.results():
                results.append(
                    {
                        "title": paper.title,
                        "authors": [author.name for author in paper.authors],
                        "summary": paper.summary,
                        "url": paper.entry_id,
                        "pdf_url": paper.pdf_url,
                        "published": str(paper.published),
                        "categories": paper.categories,
                        "source": "arxiv",
                    }
                )

            return results

        except arxiv.UnexpectedEmptyPageError as e:
            logger.error(f"ArXiv search returned empty page: {e}", exc_info=True)
            return []
        except arxiv.HTTPError as e:
            logger.error(f"ArXiv API HTTP error: {e}", exc_info=True)
            return []
        except requests.RequestException as e:
            logger.error(f"ArXiv API network error: {e}", exc_info=True)
            return []
        except ValueError as e:
            logger.error(f"ArXiv search validation error: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"ArXiv search unexpected error: {e}", exc_info=True)
            return []

    def as_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="arxiv_search",
            description="Search ArXiv for academic papers and research on a topic. Use this for scientific and technical information.",
            func=lambda query: str(self.search(query)),
        )


class PDFProcessorTool:
    """PDF text extraction"""

    def extract_from_url(self, url: str) -> dict[str, Any]:
        """
        Download and extract text from a PDF URL.

        Args:
            url: URL of the PDF

        Returns:
            Dictionary with extracted content
        """
        try:
            # Validate and sanitize URL
            try:
                url = validate_url(url)
            except ValidationError as e:
                logger.error(f"Invalid PDF URL: {e}")
                return {
                    "url": url,
                    "content": "",
                    "page_count": 0,
                    "source": "pdf",
                    "success": False,
                    "error": str(e),
                }

            headers: dict[str, str] = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response: requests.Response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            pdf_file: io.BytesIO = io.BytesIO(response.content)
            pdf_reader: PyPDF2.PdfReader = PyPDF2.PdfReader(pdf_file)

            text: str = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return {
                "url": url,
                "content": text[:20000],  # Limit to 20k chars
                "page_count": len(pdf_reader.pages),
                "source": "pdf",
                "success": True,
            }

        except requests.Timeout as e:
            logger.error(f"PDF download timeout for {url}: {e}")
            return {
                "url": url,
                "content": "",
                "page_count": 0,
                "source": "pdf",
                "success": False,
                "error": f"Timeout: {str(e)}",
            }
        except requests.ConnectionError as e:
            logger.error(f"PDF download connection error for {url}: {e}")
            return {
                "url": url,
                "content": "",
                "page_count": 0,
                "source": "pdf",
                "success": False,
                "error": f"Connection error: {str(e)}",
            }
        except requests.HTTPError as e:
            logger.error(f"PDF download HTTP error for {url}: {e}")
            return {
                "url": url,
                "content": "",
                "page_count": 0,
                "source": "pdf",
                "success": False,
                "error": f"HTTP {e.response.status_code if hasattr(e, 'response') else 'error'}: {str(e)}",
            }
        except requests.RequestException as e:
            logger.error(f"PDF download network error for {url}: {e}")
            return {
                "url": url,
                "content": "",
                "page_count": 0,
                "source": "pdf",
                "success": False,
                "error": str(e),
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid input for PDF {url}: {e}")
            return {
                "url": url,
                "content": "",
                "page_count": 0,
                "source": "pdf",
                "success": False,
                "error": str(e),
            }
        except Exception as pdf_error:
            # Check if it's a PyPDF2-specific error (works with both old and new PyPDF2 versions)
            error_name = type(pdf_error).__name__
            if "PdfReadError" in error_name or "PdfStreamError" in error_name:
                logger.error(f"PDF processing error for {url}: {pdf_error}")
                return {
                    "url": url,
                    "content": "",
                    "page_count": 0,
                    "source": "pdf",
                    "success": False,
                    "error": f"PDF processing error: {str(pdf_error)}",
                }
            # For other unexpected errors
            logger.warning(f"Unexpected PDF extraction error for {url}: {pdf_error}")
            return {
                "url": url,
                "content": "",
                "page_count": 0,
                "source": "pdf",
                "success": False,
                "error": str(pdf_error),
            }

    def extract_from_arxiv_papers(self, papers: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Extract text from ArXiv papers.

        Args:
            papers: List of paper metadata from ArXiv search

        Returns:
            List of extracted content
        """
        results: list[dict[str, Any]] = []
        for paper in papers:
            if "pdf_url" in paper:
                content: dict[str, Any] = self.extract_from_url(paper["pdf_url"])
                if content["success"]:
                    content["title"] = paper.get("title", "")
                    content["authors"] = paper.get("authors", [])
                    results.append(content)

        return results

    def as_tool(self) -> Tool:
        """Convert to LangChain Tool"""
        return Tool(
            name="pdf_processor",
            description="Extract text content from PDF documents. Use this to process academic papers and documents.",
            func=lambda url: str(self.extract_from_url(url)),
        )


class ToolManager:
    """Manager class to initialize and access all tools"""

    def __init__(self) -> None:
        self.tavily: TavilySearchTool = TavilySearchTool()
        self.scraper: WebScraperTool = WebScraperTool()
        self.arxiv: ArxivSearchTool = ArxivSearchTool()
        self.pdf_processor: PDFProcessorTool = PDFProcessorTool()

    def get_all_tools(self) -> list[Tool]:
        """Get all tools as LangChain Tool objects"""
        return [
            self.tavily.as_tool(),
            self.scraper.as_tool(),
            self.arxiv.as_tool(),
            self.pdf_processor.as_tool(),
        ]

    def get_tool_descriptions(self) -> str:
        """Get formatted descriptions of all available tools"""
        tools: list[Tool] = self.get_all_tools()
        descriptions: list[str] = []
        for tool in tools:
            descriptions.append(f"- {tool.name}: {tool.description}")
        return "\n".join(descriptions)
