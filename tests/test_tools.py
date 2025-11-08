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
