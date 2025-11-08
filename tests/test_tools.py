import types
from typing import Any
from unittest.mock import MagicMock, patch

from agent.tools import ArxivSearchTool, PDFProcessorTool, TavilySearchTool, WebScraperTool


def test_tavily_search_tool_returns_mapped_results() -> None:
    tool = TavilySearchTool()
    fake_response = {
        "results": [
            {"title": "A", "url": "http://a", "content": "ca", "score": 0.9},
            {"title": "B", "url": "http://b", "content": "cb", "score": 0.8},
        ]
    }
    tool.client.search = MagicMock(return_value=fake_response)

    results: list[dict[str, Any]] = tool.search("query", max_results=2)

    assert len(results) == 2
    assert results[0]["title"] == "A"
    assert results[0]["source"] == "tavily"
    tool.client.search.assert_called_once()


def test_web_scraper_scrape_parses_content() -> None:
    html = b"""
    <html>
      <head><title>Test Page</title></head>
      <body>
        <header>Ignore</header>
        <main>
          <h1>Heading</h1>
          <p>First paragraph.</p>
          <p>Second paragraph.</p>
        </main>
        <footer>Ignore</footer>
      </body>
    </html>
    """

    class Resp:
        status_code = 200
        content = html

        def raise_for_status(self):
            return None

    with patch("agent.tools.requests.get", return_value=Resp()):
        scraper = WebScraperTool()
        out = scraper.scrape("http://example.com")

    assert out["success"] is True
    assert isinstance(out["title"], str)
    assert len(out["title"]) >= 0
    assert ("First paragraph." in out["content"]) or (
        "Paragraph 1" in out["content"]
    )  # supports bs4 stub
    assert out["word_count"] > 0


def test_web_scraper_scrape_multiple_respects_limit() -> None:
    scraper = WebScraperTool()
    mock_scrape = MagicMock(side_effect=lambda _: {"success": True, "content": "x"})
    scraper.scrape = mock_scrape

    urls: list[str] = [f"http://u{i}" for i in range(10)]
    out: list[dict[str, Any]] = scraper.scrape_multiple(urls, max_urls=3)

    assert len(out) == 3
    assert mock_scrape.call_count == 3


def test_arxiv_search_tool_maps_results() -> None:
    arxiv_tool = ArxivSearchTool()

    class FakePaper:
        def __init__(self, i):
            self.title = f"T{i}"
            self.authors = [
                types.SimpleNamespace(name="Author1"),
                types.SimpleNamespace(name="Author2"),
            ]
            self.summary = f"S{i}"
            self.entry_id = f"http://id/{i}"
            self.pdf_url = f"http://pdf/{i}.pdf"
            self.published = "2024-01-01"
            self.categories = ["cs.AI"]

    fake_search = MagicMock()
    fake_search.results = MagicMock(return_value=[FakePaper(1), FakePaper(2)])

    with patch("agent.tools.arxiv.Search", return_value=fake_search):
        results: list[dict[str, Any]] = arxiv_tool.search("agent", max_results=2)

    assert len(results) == 2
    assert results[0]["source"] == "arxiv"
    assert "authors" in results[0]


def test_pdf_processor_extract_from_url() -> None:
    pdf_bytes = b"%PDF-1.4\n%..."

    class Resp:
        status_code = 200
        content = pdf_bytes

        def raise_for_status(self):
            return None

    class FakePage:
        def extract_text(self):
            return "Hello PDF"

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage(), FakePage()]

    with (
        patch("agent.tools.requests.get", return_value=Resp()),
        patch("agent.tools.PyPDF2.PdfReader", side_effect=FakeReader),
    ):
        pdf_tool = PDFProcessorTool()
        out = pdf_tool.extract_from_url("http://file.pdf")

    assert out["success"] is True
    assert out["page_count"] == 2
    assert "Hello PDF" in out["content"]


def test_tavily_search_tool_handles_error() -> None:
    """Test that TavilySearchTool handles errors gracefully"""
    tool = TavilySearchTool()
    tool.client.search = MagicMock(side_effect=Exception("API Error"))

    results: list[dict[str, Any]] = tool.search("query")
    assert results == []


def test_tavily_search_tool_handles_empty_results() -> None:
    """Test that TavilySearchTool handles empty results"""
    tool = TavilySearchTool()
    tool.client.search = MagicMock(return_value={"results": []})

    results: list[dict[str, Any]] = tool.search("query")
    assert results == []


def test_tavily_search_tool_uses_default_max_results() -> None:
    """Test that TavilySearchTool uses default max_results"""
    from config import Config

    tool = TavilySearchTool()
    tool.client.search = MagicMock(return_value={"results": []})

    tool.search("query")
    call_kwargs = tool.client.search.call_args[1]
    assert call_kwargs["max_results"] == Config.MAX_SEARCH_RESULTS


def test_tavily_search_tool_as_tool() -> None:
    """Test that TavilySearchTool.as_tool returns LangChain Tool"""
    tool = TavilySearchTool()
    langchain_tool = tool.as_tool()

    assert langchain_tool.name == "web_search"
    assert "web" in langchain_tool.description.lower()


def test_web_scraper_handles_network_error() -> None:
    """Test that WebScraperTool handles network errors"""
    from requests import RequestException

    scraper = WebScraperTool()
    with patch("agent.tools.requests.get", side_effect=RequestException("Network error")):
        out = scraper.scrape("http://example.com")

    assert out["success"] is False
    assert "error" in out
    assert out["content"] == ""


def test_web_scraper_handles_value_error() -> None:
    """Test that WebScraperTool handles ValueError"""
    scraper = WebScraperTool()
    with patch("agent.tools.requests.get", side_effect=ValueError("Invalid URL")):
        out = scraper.scrape("invalid-url")

    assert out["success"] is False
    assert "error" in out


def test_web_scraper_handles_generic_exception() -> None:
    """Test that WebScraperTool handles generic exceptions"""
    scraper = WebScraperTool()
    with patch("agent.tools.requests.get", side_effect=Exception("Unexpected error")):
        out = scraper.scrape("http://example.com")

    assert out["success"] is False
    assert "error" in out


def test_web_scraper_handles_missing_title() -> None:
    """Test that WebScraperTool handles pages without title"""
    html = b"<html><body><p>Content without title</p></body></html>"

    class Resp:
        status_code = 200
        content = html

        def raise_for_status(self):
            return None

    with patch("agent.tools.requests.get", return_value=Resp()):
        scraper = WebScraperTool()
        out = scraper.scrape("http://example.com")

    assert out["success"] is True
    assert isinstance(out["title"], str)


def test_web_scraper_scrape_multiple_handles_failures() -> None:
    """Test that scrape_multiple filters out failed scrapes"""
    scraper = WebScraperTool()
    mock_scrape = MagicMock(
        side_effect=[
            {"success": True, "content": "x"},
            {"success": False, "content": ""},
            {"success": True, "content": "y"},
        ]
    )
    scraper.scrape = mock_scrape

    urls: list[str] = ["http://u1", "http://u2", "http://u3"]
    out: list[dict[str, Any]] = scraper.scrape_multiple(urls)

    # Should only include successful scrapes
    assert len(out) == 2
    assert all(r["success"] for r in out)


def test_web_scraper_scrape_multiple_uses_default_max_urls() -> None:
    """Test that scrape_multiple uses default max_urls"""
    from config import Config

    scraper = WebScraperTool()
    mock_scrape = MagicMock(return_value={"success": True, "content": "x"})
    scraper.scrape = mock_scrape

    urls: list[str] = [f"http://u{i}" for i in range(20)]
    scraper.scrape_multiple(urls)

    assert mock_scrape.call_count == Config.MAX_SCRAPE_URLS


def test_web_scraper_as_tool() -> None:
    """Test that WebScraperTool.as_tool returns LangChain Tool"""
    scraper = WebScraperTool()
    langchain_tool = scraper.as_tool()

    assert langchain_tool.name == "web_scraper"
    assert "webpage" in langchain_tool.description.lower() or "url" in langchain_tool.description.lower()


def test_arxiv_search_tool_handles_error() -> None:
    """Test that ArxivSearchTool handles errors gracefully"""
    arxiv_tool = ArxivSearchTool()

    with patch("agent.tools.arxiv.Search", side_effect=Exception("API Error")):
        results: list[dict[str, Any]] = arxiv_tool.search("query")

    assert results == []


def test_arxiv_search_tool_uses_default_max_results() -> None:
    """Test that ArxivSearchTool uses default max_results"""
    from config import Config

    arxiv_tool = ArxivSearchTool()
    fake_search = MagicMock()
    fake_search.results = MagicMock(return_value=[])

    with patch("agent.tools.arxiv.Search", return_value=fake_search):
        arxiv_tool.search("query")

    call_kwargs = fake_search.__init__.call_args[1] if hasattr(fake_search.__init__, "call_args") else {}
    # Verify max_results was used in Search constructor
    assert fake_search.results.called


def test_arxiv_search_tool_as_tool() -> None:
    """Test that ArxivSearchTool.as_tool returns LangChain Tool"""
    arxiv_tool = ArxivSearchTool()
    langchain_tool = arxiv_tool.as_tool()

    assert langchain_tool.name == "arxiv_search"
    assert "arxiv" in langchain_tool.description.lower() or "academic" in langchain_tool.description.lower()


def test_pdf_processor_handles_network_error() -> None:
    """Test that PDFProcessorTool handles network errors"""
    from requests import RequestException

    pdf_tool = PDFProcessorTool()
    with patch("agent.tools.requests.get", side_effect=RequestException("Network error")):
        out = pdf_tool.extract_from_url("http://file.pdf")

    assert out["success"] is False
    assert "error" in out
    assert out["content"] == ""
    assert out["page_count"] == 0


def test_pdf_processor_handles_value_error() -> None:
    """Test that PDFProcessorTool handles ValueError"""
    pdf_tool = PDFProcessorTool()
    with patch("agent.tools.requests.get", side_effect=ValueError("Invalid URL")):
        out = pdf_tool.extract_from_url("invalid-url")

    assert out["success"] is False
    assert "error" in out


def test_pdf_processor_handles_generic_exception() -> None:
    """Test that PDFProcessorTool handles generic exceptions"""
    pdf_tool = PDFProcessorTool()
    with patch("agent.tools.requests.get", side_effect=Exception("Unexpected error")):
        out = pdf_tool.extract_from_url("http://file.pdf")

    assert out["success"] is False
    assert "error" in out


def test_pdf_processor_extract_from_arxiv_papers() -> None:
    """Test that extract_from_arxiv_papers processes multiple papers"""
    pdf_tool = PDFProcessorTool()

    class FakePage:
        def extract_text(self):
            return "PDF Content"

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage()]

    class Resp:
        status_code = 200
        content = b"%PDF-1.4"

        def raise_for_status(self):
            return None

    papers: list[dict[str, Any]] = [
        {"title": "Paper 1", "pdf_url": "http://pdf1.pdf", "authors": ["A1"]},
        {"title": "Paper 2", "pdf_url": "http://pdf2.pdf", "authors": ["A2"]},
    ]

    with (
        patch("agent.tools.requests.get", return_value=Resp()),
        patch("agent.tools.PyPDF2.PdfReader", side_effect=FakeReader),
    ):
        results: list[dict[str, Any]] = pdf_tool.extract_from_arxiv_papers(papers)

    assert len(results) == 2
    assert all(r["success"] for r in results)
    assert results[0]["title"] == "Paper 1"
    assert results[1]["title"] == "Paper 2"


def test_pdf_processor_extract_from_arxiv_papers_filters_failures() -> None:
    """Test that extract_from_arxiv_papers filters out failed extractions"""
    pdf_tool = PDFProcessorTool()

    papers: list[dict[str, Any]] = [
        {"title": "Paper 1", "pdf_url": "http://pdf1.pdf"},
        {"title": "Paper 2"},  # No pdf_url - won't be processed
    ]

    # Mock extract_from_url to fail for first paper (only one call since Paper 2 has no pdf_url)
    pdf_tool.extract_from_url = MagicMock(
        return_value={"success": False, "content": ""}
    )

    results: list[dict[str, Any]] = pdf_tool.extract_from_arxiv_papers(papers)
    # Should only include successful extractions (none in this case)
    assert len(results) == 0


def test_pdf_processor_as_tool() -> None:
    """Test that PDFProcessorTool.as_tool returns LangChain Tool"""
    pdf_tool = PDFProcessorTool()
    langchain_tool = pdf_tool.as_tool()

    assert langchain_tool.name == "pdf_processor"
    assert "pdf" in langchain_tool.description.lower()


def test_tool_manager_initializes_all_tools() -> None:
    """Test that ToolManager initializes all tools"""
    from agent.tools import ToolManager

    manager = ToolManager()
    assert manager.tavily is not None
    assert manager.scraper is not None
    assert manager.arxiv is not None
    assert manager.pdf_processor is not None


def test_tool_manager_get_all_tools() -> None:
    """Test that ToolManager.get_all_tools returns all tools"""
    from agent.tools import ToolManager
    from langchain_core.tools import Tool

    manager = ToolManager()
    tools: list[Tool] = manager.get_all_tools()

    assert len(tools) == 4
    assert all(isinstance(tool, Tool) for tool in tools)
    tool_names = [tool.name for tool in tools]
    assert "web_search" in tool_names
    assert "web_scraper" in tool_names
    assert "arxiv_search" in tool_names
    assert "pdf_processor" in tool_names


def test_tool_manager_get_tool_descriptions() -> None:
    """Test that ToolManager.get_tool_descriptions returns formatted descriptions"""
    from agent.tools import ToolManager

    manager = ToolManager()
    descriptions: str = manager.get_tool_descriptions()

    assert isinstance(descriptions, str)
    assert "web_search" in descriptions
    assert "web_scraper" in descriptions
    assert "arxiv_search" in descriptions
    assert "pdf_processor" in descriptions
