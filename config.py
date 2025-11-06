"""
Configuration and API clients for Autonomous Research Assistant
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Optional

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the research assistant"""
    
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    
    # LangSmith Tracing
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "true")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "GenAIAppWithOpenAI")
    
    # Model Configuration
    MODEL_NAME: str = "llama-3.3-70b-versatile"  # Fast and capable Groq model (updated)
    MODEL_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 8192
    
    # RAG Configuration
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    TOP_K_RESULTS: int = 5
    
    # ChromaDB Configuration
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "research_documents"
    
    # Search Configuration
    MAX_SEARCH_RESULTS: int = 5
    MAX_ARXIV_RESULTS: int = 3
    MAX_SCRAPE_URLS: int = 5
    
    # Agent Configuration
    MAX_ITERATIONS: int = 10
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required API keys are present"""
        required_keys = {
            "GROQ_API_KEY": cls.GROQ_API_KEY,
            "TAVILY_API_KEY": cls.TAVILY_API_KEY,
        }
        
        missing_keys = [key for key, value in required_keys.items() if not value]
        
        if missing_keys:
            raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
        
        return True


def get_llm(temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> ChatGroq:
    """
    Get configured Groq LLM instance.
    
    Args:
        temperature: Override default temperature
        max_tokens: Override default max tokens
        
    Returns:
        Configured ChatGroq instance
    """
    Config.validate()
    
    return ChatGroq(
        groq_api_key=Config.GROQ_API_KEY,
        model_name=Config.MODEL_NAME,
        temperature=temperature or Config.MODEL_TEMPERATURE,
        max_tokens=max_tokens or Config.MAX_TOKENS,
    )


# Set LangSmith environment variables for tracing
if Config.LANGSMITH_TRACING.lower() == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = Config.LANGSMITH_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = Config.LANGSMITH_API_KEY or Config.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = Config.LANGSMITH_PROJECT

