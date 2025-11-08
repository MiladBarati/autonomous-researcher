import importlib.machinery
import importlib.util
import os
import sys
import types
from collections.abc import Callable
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
if "requests" not in sys.modules and importlib.util.find_spec("requests") is None:

    class _Response:
        def __init__(self) -> None:
            self.status_code = 200
            self.content = b""
            self.text = ""

        def raise_for_status(self) -> None:
            return None

    class _RequestException(Exception):
        pass

    requests_mod = types.ModuleType("requests")
    requests_mod.get = lambda *_args, **_kwargs: _Response()  # type: ignore[attr-defined]
    requests_mod.RequestException = _RequestException  # type: ignore[attr-defined]
    _ensure_module("requests", requests_mod)

# Stub: langchain_core
if "langchain_core" not in sys.modules and importlib.util.find_spec("langchain_core") is None:

    class _BaseMessage:
        def __init__(self, content: str = "") -> None:
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
        def __init__(
            self,
            name: str = "",
            description: str = "",
            func: Callable[..., Any] | None = None,
            **_kwargs: Any,
        ) -> None:
            self.name = name
            self.description = description
            self.func = func

    lc_core_tools_mod = types.ModuleType("langchain_core.tools")
    lc_core_tools_mod.Tool = _Tool  # type: ignore[attr-defined]
    _ensure_module("langchain_core.tools", lc_core_tools_mod)

# Stub: langchain_groq
if "langchain_groq" not in sys.modules and importlib.util.find_spec("langchain_groq") is None:

    class _ChatGroq:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def invoke(self, _messages: Any) -> Any:
            return type("Response", (), {"content": "stub response"})

    lc_groq_mod = types.ModuleType("langchain_groq")
    lc_groq_mod.ChatGroq = _ChatGroq  # type: ignore[attr-defined]
    _ensure_module("langchain_groq", lc_groq_mod)

# Stub: pydantic
if "pydantic" not in sys.modules and importlib.util.find_spec("pydantic") is None:

    class _SecretStr:
        def __init__(self, value: Any) -> None:
            self._value = value

        def __str__(self) -> str:
            return str(self._value)

        def get_secret_value(self) -> Any:
            return self._value

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.SecretStr = _SecretStr  # type: ignore[attr-defined]
    _ensure_module("pydantic", pydantic_mod)

# Stub: langgraph
if "langgraph" not in sys.modules and importlib.util.find_spec("langgraph") is None:

    class _StateGraph:
        def __init__(self, _state: Any) -> None:
            self._nodes: dict[str, Callable[..., Any]] = {}
            self._entry_point: str | None = None
            self._edges: list[tuple[Any, Any]] = []

        def add_node(self, name: str, func: Callable[..., Any]) -> "_StateGraph":
            self._nodes[name] = func
            return self

        def set_entry_point(self, name: str) -> "_StateGraph":
            self._entry_point = name
            return self

        def add_edge(self, from_node: Any, to_node: Any) -> "_StateGraph":
            self._edges.append((from_node, to_node))
            return self

        def compile(self) -> Any:
            class _CompiledGraph:
                def __init__(self, graph_builder: "_StateGraph") -> None:
                    self._graph_builder = graph_builder
                    self._nodes = graph_builder._nodes
                    self._entry_point = graph_builder._entry_point
                    self._edges = graph_builder._edges

                def invoke(self, state: Any) -> Any:
                    # Execute nodes in sequence for stub
                    if not self._entry_point:
                        return state

                    current_node: str | None = self._entry_point
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
                            next_node: str | None = None
                            for from_node, to_node in self._edges:
                                if from_node == current_node:
                                    # Check if to_node is END - compare by type/name since it's a class
                                    # END is a special marker class, check various ways
                                    is_end = (
                                        to_node is None
                                        or (
                                            hasattr(to_node, "__name__")
                                            and to_node.__name__ == "END"
                                        )
                                        or (type(to_node).__name__ == "END")
                                        or str(to_node) == "END"
                                        or (
                                            hasattr(to_node, "__class__")
                                            and to_node.__class__.__name__ == "END"
                                        )
                                    )
                                    if is_end:
                                        return state
                                    next_node = to_node if isinstance(to_node, str) else None
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
if "arxiv" not in sys.modules and importlib.util.find_spec("arxiv") is None:

    class _ArxivSearch:
        SortCriterion = type("SortCriterion", (), {"Relevance": object()})

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def results(self) -> list[Any]:  # empty default, tests patch this
            return []

    mod = types.ModuleType("arxiv")
    mod.Search = _ArxivSearch  # type: ignore[attr-defined]
    mod.SortCriterion = _ArxivSearch.SortCriterion  # type: ignore[attr-defined]
    _ensure_module("arxiv", mod)


# Stub: tavily
if "tavily" not in sys.modules and importlib.util.find_spec("tavily") is None:

    class _TavilyClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def search(self, *_args: Any, **_kwargs: Any) -> dict[str, list[Any]]:
            return {"results": []}

    mod = types.ModuleType("tavily")
    mod.TavilyClient = _TavilyClient  # type: ignore[attr-defined]
    _ensure_module("tavily", mod)


# Stub: sentence_transformers
if (
    "sentence_transformers" not in sys.modules
    and importlib.util.find_spec("sentence_transformers") is None
):

    class _SentenceTransformer:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def encode(self, texts: Any, **_kwargs: Any) -> list[list[float]]:
            return [[0.0] * 3 for _ in (texts if isinstance(texts, list) else [texts])]

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _SentenceTransformer  # type: ignore[attr-defined]
    _ensure_module("sentence_transformers", mod)


# Stub: chromadb
if "chromadb" not in sys.modules and importlib.util.find_spec("chromadb") is None:

    class _FakeCollection:
        def add(self, **_kwargs: Any) -> None:
            return None

        def query(self, **_kwargs: Any) -> dict[str, list[list[Any]]]:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        def count(self) -> int:
            return 0

    class _PersistentClient:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._col: _FakeCollection | None = None

        def get_collection(self, _name: str) -> _FakeCollection:
            if self._col is None:
                raise Exception("no collection")
            return self._col

        def create_collection(self, _name: str, _metadata: Any = None) -> _FakeCollection:
            self._col = _FakeCollection()
            return self._col

        def delete_collection(self, _name: str) -> None:
            self._col = None

    class _Settings:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    chromadb_mod = types.ModuleType("chromadb")
    chromadb_mod.PersistentClient = _PersistentClient  # type: ignore[attr-defined]
    chromadb_mod.Collection = _FakeCollection  # type: ignore[attr-defined]
    _ensure_module("chromadb", chromadb_mod)
    chromadb_config_mod = types.ModuleType("chromadb.config")
    chromadb_config_mod.Settings = _Settings  # type: ignore[attr-defined]
    _ensure_module("chromadb.config", chromadb_config_mod)


# Stub: PyPDF2
if "PyPDF2" not in sys.modules and importlib.util.find_spec("PyPDF2") is None:

    class _PdfReader:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.pages: list[Any] = []

    mod = types.ModuleType("PyPDF2")
    mod.PdfReader = _PdfReader  # type: ignore[attr-defined]
    _ensure_module("PyPDF2", mod)


# Stub: bs4 (BeautifulSoup)
if "bs4" not in sys.modules and importlib.util.find_spec("bs4") is None:

    class _Tag:
        def __init__(self, text: str = "") -> None:
            self._text = text

        def get_text(self) -> str:
            return self._text

        def decompose(self) -> None:
            pass

    class _Soup:
        def __init__(self, content: Any, _parser: Any) -> None:
            self._content = content

        def find(self, name: str) -> _Tag | None:
            if name == "title":
                return _Tag("Stub Title")
            if name in ("main", "article", "body"):
                return self
            return None

        def find_all(self, _names: Any) -> list[_Tag]:
            return [_Tag("Paragraph 1"), _Tag("Paragraph 2")]

        def __call__(self, *_args: Any, **_kwargs: Any) -> list[Any]:
            return []

        def __getitem__(self, item: Any) -> Any:
            return None

        def get_text(self) -> str:
            return "Stub text"

    bs4_mod = types.ModuleType("bs4")
    bs4_mod.BeautifulSoup = _Soup  # type: ignore[attr-defined]
    _ensure_module("bs4", bs4_mod)


# Stub: langchain_text_splitters and legacy langchain.text_splitter
if (
    "langchain_text_splitters" not in sys.modules
    and importlib.util.find_spec("langchain_text_splitters") is None
):

    class _RecursiveCharacterTextSplitter:
        def __init__(
            self,
            chunk_size: int = 1000,
            chunk_overlap: int = 200,
            length_function: Callable[[str], int] = len,
            separators: list[str] | None = None,
            _separators: list[str] | None = None,
        ) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            self.length_function = length_function
            self.separators = separators or _separators

        def split_text(self, text: str) -> list[str]:
            if not text:
                return []
            n = self.chunk_size
            o = self.chunk_overlap
            chunks: list[str] = []
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
