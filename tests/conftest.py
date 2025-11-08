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


# IMPORTANT: Stub langchain modules FIRST since agent/__init__.py imports agent.graph
# which imports these at module level

# Stub: requests (used by tools.py)
if "requests" not in sys.modules:
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        class _Response:
            def __init__(self):
                self.status_code = 200
                self.content = b""
                self.text = ""

            def raise_for_status(self):
                return None

        class _RequestException(Exception):
            pass

        requests_mod = types.ModuleType("requests")
        requests_mod.get = lambda *args, **kwargs: _Response()  # type: ignore[assignment]
        requests_mod.RequestException = _RequestException  # type: ignore[attr-defined]
        _ensure_module("requests", requests_mod)

# Stub: langchain_core
if "langchain_core" not in sys.modules:
    try:
        import langchain_core  # type: ignore[import-untyped]
    except ImportError:
        class _BaseMessage:
            def __init__(self, content=""):
                self.content = content

        class _AIMessage(_BaseMessage):
            pass

        class _HumanMessage(_BaseMessage):
            pass

        class _SystemMessage(_BaseMessage):
            pass

        lc_core_mod = types.ModuleType("langchain_core")
        lc_core_messages_mod = types.ModuleType("langchain_core.messages")
        lc_core_messages_mod.AIMessage = _AIMessage  # type: ignore[attr-defined]
        lc_core_messages_mod.HumanMessage = _HumanMessage  # type: ignore[attr-defined]
        lc_core_messages_mod.SystemMessage = _SystemMessage  # type: ignore[attr-defined]
        lc_core_messages_mod.BaseMessage = _BaseMessage  # type: ignore[attr-defined]
        _ensure_module("langchain_core", lc_core_mod)
        _ensure_module("langchain_core.messages", lc_core_messages_mod)

        # Stub: langchain_core.tools
        class _Tool:
            def __init__(self, name="", description="", func=None, **kwargs):
                self.name = name
                self.description = description
                self.func = func

        lc_core_tools_mod = types.ModuleType("langchain_core.tools")
        lc_core_tools_mod.Tool = _Tool  # type: ignore[attr-defined]
        _ensure_module("langchain_core.tools", lc_core_tools_mod)

# Stub: langchain_groq
if "langchain_groq" not in sys.modules:
    try:
        import langchain_groq  # type: ignore[import-untyped]
    except ImportError:
        class _ChatGroq:
            def __init__(self, *_args, **_kwargs):
                pass

            def invoke(self, _messages):
                return type("Response", (), {"content": "stub response"})

        lc_groq_mod = types.ModuleType("langchain_groq")
        lc_groq_mod.ChatGroq = _ChatGroq  # type: ignore[attr-defined]
        _ensure_module("langchain_groq", lc_groq_mod)

# Stub: pydantic
if "pydantic" not in sys.modules:
    try:
        import pydantic  # type: ignore[import-untyped]
    except ImportError:
        class _SecretStr:
            def __init__(self, value):
                self._value = value

            def __str__(self):
                return str(self._value)

            def get_secret_value(self):
                return self._value

        pydantic_mod = types.ModuleType("pydantic")
        pydantic_mod.SecretStr = _SecretStr  # type: ignore[attr-defined]
        _ensure_module("pydantic", pydantic_mod)

# Stub: langgraph
if "langgraph" not in sys.modules:
    try:
        import langgraph  # type: ignore[import-untyped]
    except ImportError:
        class _StateGraph:
            def __init__(self, _state):
                self._nodes = {}
                self._entry_point = None
                self._edges = []

            def add_node(self, name, func):
                self._nodes[name] = func
                return self

            def set_entry_point(self, name):
                self._entry_point = name
                return self

            def add_edge(self, from_node, to_node):
                self._edges.append((from_node, to_node))
                return self

            def compile(self):
                class _CompiledGraph:
                    def __init__(self, graph_builder):
                        self._graph_builder = graph_builder
                        self._nodes = graph_builder._nodes
                        self._entry_point = graph_builder._entry_point
                        self._edges = graph_builder._edges
                    
                    def invoke(self, state):
                        # Execute nodes in sequence for stub
                        if not self._entry_point:
                            return state
                        
                        current_node = self._entry_point
                        max_iterations = 20  # Prevent infinite loops
                        iteration = 0
                        
                        while current_node and iteration < max_iterations:
                            if current_node in self._nodes:
                                node_func = self._nodes[current_node]
                                # Nodes return partial state updates - merge with existing state
                                state_update = node_func(state)
                                if isinstance(state_update, dict):
                                    # Merge update into existing state (LangGraph behavior)
                                    state = {**state, **state_update}
                                else:
                                    state = state_update
                                iteration += 1
                                
                                # Find next node
                                next_node = None
                                for from_node, to_node in self._edges:
                                    if from_node == current_node:
                                        # Check if to_node is END - compare by type/name since it's a class
                                        # END is a special marker class, check various ways
                                        is_end = (
                                            to_node is None or
                                            (hasattr(to_node, '__name__') and to_node.__name__ == 'END') or
                                            (type(to_node).__name__ == 'END') or
                                            str(to_node) == 'END' or
                                            (hasattr(to_node, '__class__') and to_node.__class__.__name__ == 'END')
                                        )
                                        if is_end:
                                            return state
                                        next_node = to_node
                                        break
                                
                                current_node = next_node
                            else:
                                break
                        
                        return state
                return _CompiledGraph(self)

        class _END:
            pass

        lg_mod = types.ModuleType("langgraph")
        lg_graph_mod = types.ModuleType("langgraph.graph")
        lg_graph_mod.StateGraph = _StateGraph  # type: ignore[attr-defined]
        lg_graph_mod.END = _END  # type: ignore[attr-defined]
        _ensure_module("langgraph", lg_mod)
        _ensure_module("langgraph.graph", lg_graph_mod)


# Stub: arxiv
if "arxiv" not in sys.modules:
    try:
        import arxiv  # type: ignore[import-untyped]
    except ImportError:
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
if "tavily" not in sys.modules:
    try:
        import tavily  # type: ignore[import-untyped]
    except ImportError:
        class _TavilyClient:
            def __init__(self, *_args, **_kwargs):
                pass

            def search(self, *_args, **_kwargs):
                return {"results": []}

    mod = types.ModuleType("tavily")
    mod.TavilyClient = _TavilyClient  # type: ignore[attr-defined]
    _ensure_module("tavily", mod)


# Stub: sentence_transformers
if "sentence_transformers" not in sys.modules:
    try:
        import sentence_transformers  # type: ignore[import-untyped]
    except ImportError:
        class _SentenceTransformer:
            def __init__(self, *_args, **_kwargs):
                pass

            def encode(self, texts, **_kwargs):
                return [[0.0] * 3 for _ in (texts if isinstance(texts, list) else [texts])]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    _ensure_module("sentence_transformers", mod)


# Stub: chromadb
if "chromadb" not in sys.modules:
    try:
        import chromadb  # type: ignore[import-untyped]
    except ImportError:
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
        chromadb_mod.Collection = _FakeCollection  # type: ignore[attr-defined]
        _ensure_module("chromadb", chromadb_mod)
        chromadb_config_mod = types.ModuleType("chromadb.config")
        chromadb_config_mod.Settings = _Settings  # type: ignore[attr-defined]
        _ensure_module("chromadb.config", chromadb_config_mod)


# Stub: PyPDF2
if "PyPDF2" not in sys.modules:
    try:
        import PyPDF2  # type: ignore[import-untyped]
    except ImportError:
        class _PdfReader:
            def __init__(self, *_args, **_kwargs):
                self.pages = []

    mod = types.ModuleType("PyPDF2")
    mod.PdfReader = _PdfReader  # type: ignore[attr-defined]
    _ensure_module("PyPDF2", mod)


# Stub: bs4 (BeautifulSoup)
if "bs4" not in sys.modules:
    try:
        import bs4  # type: ignore[import-untyped]
    except ImportError:
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
if "langchain_text_splitters" not in sys.modules:
    try:
        import langchain_text_splitters  # type: ignore[import-untyped]
    except ImportError:
        class _RecursiveCharacterTextSplitter:
            def __init__(
                self, chunk_size=1000, chunk_overlap=200, length_function=len, separators=None, _separators=None
            ):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.length_function = length_function or (lambda x: len(x))
                self.separators = separators or _separators

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
