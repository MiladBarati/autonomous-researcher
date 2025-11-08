import importlib.machinery
import os
import sys
import types
from typing import Any

# Ensure project root is on sys.path so tests can import `agent` package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Provide minimal stubs for optional heavy/remote dependencies if not installed


def _ensure_module(name: str, module_obj: Any) -> None:
    if isinstance(module_obj, types.ModuleType):
        module = module_obj
    else:
        module = types.ModuleType(name)
        for k, v in module_obj.__dict__.items():
            if not k.startswith("__") or k in ("__doc__",):
                setattr(module, k, v)
    if getattr(module, "__spec__", None) is None:
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    sys.modules[name] = module


# Stub: arxiv
try:
    pass
except Exception:  # pragma: no cover

    class _ArxivSearch:
        SortCriterion = type("SortCriterion", (), {"Relevance": object()})

        def __init__(self, *_args, **_kwargs):
            pass

        def results(self):  # empty default, tests patch this
            return []

    mod = types.ModuleType("arxiv")
    mod.Search = _ArxivSearch  # type: ignore[attr-defined]
    mod.SortCriterion = _ArxivSearch.SortCriterion  # type: ignore[attr-defined]
    _ensure_module("arxiv", mod)


# Stub: tavily
try:
    pass
except Exception:  # pragma: no cover

    class _TavilyClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def search(self, *_args, **_kwargs):
            return {"results": []}

    mod = types.ModuleType("tavily")
    mod.TavilyClient = _TavilyClient  # type: ignore[attr-defined]
    _ensure_module("tavily", mod)


# Stub: sentence_transformers
try:
    pass
except Exception:  # pragma: no cover

    class _SentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return [[0.0] * 3 for _ in (texts if isinstance(texts, list) else [texts])]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    _ensure_module("sentence_transformers", mod)


# Stub: chromadb
try:
    pass
except Exception:  # pragma: no cover

    class _FakeCollection:
        def add(self, **_kwargs):
            return None

        def query(self, **_kwargs):
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        def count(self):
            return 0

    class _PersistentClient:
        def __init__(self, *_args, **_kwargs):
            self._col = None

        def get_collection(self, _name):
            if self._col is None:
                raise Exception("no collection")
            return self._col

        def create_collection(self, _name, _metadata=None):
            self._col = _FakeCollection()
            return self._col

        def delete_collection(self, _name):
            self._col = None

    class _Settings:
        def __init__(self, **_kwargs):
            pass

    chromadb_mod = types.ModuleType("chromadb")
    chromadb_mod.PersistentClient = _PersistentClient  # type: ignore[attr-defined]
    _ensure_module("chromadb", chromadb_mod)
    chromadb_config_mod = types.ModuleType("chromadb.config")
    chromadb_config_mod.Settings = _Settings  # type: ignore[attr-defined]
    _ensure_module("chromadb.config", chromadb_config_mod)


# Stub: PyPDF2
try:
    pass
except Exception:  # pragma: no cover

    class _PdfReader:
        def __init__(self, *_args, **_kwargs):
            self.pages = []

    mod = types.ModuleType("PyPDF2")
    mod.PdfReader = _PdfReader  # type: ignore[attr-defined]
    _ensure_module("PyPDF2", mod)


# Stub: bs4 (BeautifulSoup)
try:
    pass
except Exception:  # pragma: no cover

    class _Tag:
        def __init__(self, text=""):
            self._text = text

        def get_text(self):
            return self._text

        def decompose(self):
            pass

    class _Soup:
        def __init__(self, content, _parser):
            self._content = content

        def find(self, name):
            if name == "title":
                return _Tag("Stub Title")
            if name in ("main", "article", "body"):
                return self
            return None

        def find_all(self, _names):
            return [_Tag("Paragraph 1"), _Tag("Paragraph 2")]

        def __call__(self, *_args, **_kwargs):
            return []

        def __getitem__(self, item):
            return None

        def get_text(self):
            return "Stub text"

    bs4_mod = types.ModuleType("bs4")
    bs4_mod.BeautifulSoup = _Soup  # type: ignore[attr-defined]
    _ensure_module("bs4", bs4_mod)


# Stub: langchain_text_splitters and legacy langchain.text_splitter
try:
    pass
except Exception:  # pragma: no cover

    class _RecursiveCharacterTextSplitter:
        def __init__(
            self, chunk_size=1000, chunk_overlap=200, length_function=len, _separators=None
        ):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function or (lambda x: len(x))

        def split_text(self, text):
            if not text:
                return []
            n = self.chunk_size
            o = self.chunk_overlap
            chunks = []
            i = 0
            while i < len(text):
                chunk = text[i : i + n]
                chunks.append(chunk)
                if n <= o:
                    break
                i += n - o
            return chunks

    lc_ts_mod = types.ModuleType("langchain_text_splitters")
    lc_ts_mod.RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter  # type: ignore[attr-defined]
    _ensure_module("langchain_text_splitters", lc_ts_mod)

    # Legacy path used in try/except fallback in code
    lc_mod = types.ModuleType("langchain")
    lc_text_splitter_mod = types.ModuleType("langchain.text_splitter")
    lc_text_splitter_mod.RecursiveCharacterTextSplitter = _RecursiveCharacterTextSplitter  # type: ignore[attr-defined]
    _ensure_module("langchain", lc_mod)
    _ensure_module("langchain.text_splitter", lc_text_splitter_mod)
