"""
Streamlit Web Interface for Autonomous Research Assistant

Provides an interactive UI for:
- Topic input
- Real-time research progress tracking
- Results visualization
- Report download
"""

from datetime import datetime

import streamlit as st

from agent.graph import create_research_graph
from agent.state import ResearchState
from config import Config

# Page configuration
st.set_page_config(
    page_title="Autonomous Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables"""
    if "research_results" not in st.session_state:
        st.session_state.research_results = None
    if "research_history" not in st.session_state:
        st.session_state.research_history = []
    if "agent" not in st.session_state:
        st.session_state.agent = None


def display_header() -> None:
    """Display application header"""
    st.title("🔬 Autonomous Research Assistant")
    st.markdown(
        """
    An AI-powered research agent that autonomously gathers, analyzes, and synthesizes information using:
    - **LangGraph** for workflow orchestration
    - **LangChain** for tool integration
    - **Groq** for fast LLM inference
    - **RAG** (Retrieval-Augmented Generation) for context-aware synthesis
    """
    )
    st.divider()


def display_sidebar() -> None:
    """Display sidebar with settings and info"""
    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Model Settings")
        st.info(
            f"""
        - **Model**: {Config.MODEL_NAME}
        - **Temperature**: {Config.MODEL_TEMPERATURE}
        - **Max Tokens**: {Config.MAX_TOKENS}
        """
        )

        with st.expander("Alternative Models"):
            st.markdown(
                """
            If you encounter model errors, update `MODEL_NAME` in `config.py`:
            - `llama-3.3-70b-versatile` (current default)
            - `llama-3.1-8b-instant` (faster, smaller)
            - `mixtral-8x7b-32768` (alternative large model)
            - `gemma2-9b-it` (efficient)

            See [Groq Models](https://console.groq.com/docs/models) for all available models.
            """
            )

        st.subheader("RAG Settings")
        st.info(
            f"""
        - **Chunk Size**: {Config.CHUNK_SIZE}
        - **Embedding Model**: {Config.EMBEDDING_MODEL}
        - **Top-K Results**: {Config.TOP_K_RESULTS}
        """
        )

        st.subheader("Search Settings")
        st.info(
            f"""
        - **Max Web Results**: {Config.MAX_SEARCH_RESULTS}
        - **Max ArXiv Papers**: {Config.MAX_ARXIV_RESULTS}
        - **Max URLs to Scrape**: {Config.MAX_SCRAPE_URLS}
        """
        )

        st.divider()

        st.subheader("📚 Research History")
        if st.session_state.research_history:
            for i, item in enumerate(reversed(st.session_state.research_history[-5:]), 1):
                with st.expander(f"{i}. {item['topic'][:50]}..."):
                    st.text(f"Date: {item['timestamp']}")
                    st.text(f"Status: {item['status']}")
        else:
            st.text("No research history yet")


def display_research_input() -> tuple[str, bool]:
    """Display research topic input form"""
    st.header("🎯 Start New Research")

    col1, col2 = st.columns([3, 1])

    with col1:
        topic = st.text_input(
            "Research Topic",
            placeholder="e.g., Latest developments in quantum computing, Climate change solutions, AI ethics...",
            help="Enter any topic you want to research. The agent will autonomously gather and synthesize information.",
        )

    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        start_button = st.button("🚀 Start Research", type="primary", use_container_width=True)

    return topic, start_button


def display_progress_tracking(state: ResearchState) -> None:
    """Display real-time progress of research"""
    st.subheader("📊 Research Progress")

    status = state.get("status", "initialized")
    step_count = state.get("step_count", 0)

    # Progress stages
    stages = {
        "initialized": 0,
        "planned": 1,
        "searched": 2,
        "scraped": 3,
        "arxiv_searched": 4,
        "documents_processed": 5,
        "stored": 6,
        "completed": 7,
    }

    current_stage = stages.get(status, 0)
    progress = (current_stage / 7) * 100

    st.progress(progress / 100)

    # Status messages
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Status", status.replace("_", " ").title())
    with col2:
        st.metric("Steps Completed", step_count)
    with col3:
        st.metric("Progress", f"{int(progress)}%")

    # Stage indicators
    st.markdown("### Pipeline Stages")
    stages_list = [
        ("📋 Planning", "planned"),
        ("🔍 Web Search", "searched"),
        ("📄 Content Scraping", "scraped"),
        ("🎓 ArXiv Search", "arxiv_searched"),
        ("📚 Document Processing", "documents_processed"),
        ("💾 Vector Storage", "stored"),
        ("✨ Synthesis", "completed"),
    ]

    cols = st.columns(7)
    for i, (stage_name, stage_key) in enumerate(stages_list):
        with cols[i]:
            if stages.get(status, 0) > stages[stage_key]:
                st.success(stage_name)
            elif status == stage_key:
                st.info(f"⏳ {stage_name}")
            else:
                st.text(stage_name)


def display_research_results(state: ResearchState) -> None:
    """Display research results with expandable sections"""
    st.header("📝 Research Results")

    # Main synthesis
    synthesis = state.get("synthesis")
    if synthesis:
        st.subheader("📊 Research Report")
        st.markdown(synthesis)

        # Download button
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"research_report_{timestamp}.md"
        st.download_button(
            label="📥 Download Report",
            data=synthesis,
            file_name=filename,
            mime="text/markdown",
            use_container_width=True,
        )

    st.divider()

    # Expandable sections for details
    col1, col2 = st.columns(2)

    with col1:
        # Research Plan
        if state.get("research_plan"):
            with st.expander("📋 Research Plan", expanded=False):
                st.markdown(state["research_plan"])

        # Search Results
        if state.get("search_results"):
            with st.expander(
                f"🔍 Web Search Results ({len(state['search_results'])})", expanded=False
            ):
                for i, result in enumerate(state["search_results"][:10], 1):
                    st.markdown(f"**{i}. {result.get('title', 'No title')}**")
                    st.text(f"URL: {result.get('url', 'N/A')}")
                    st.text(f"Relevance Score: {result.get('score', 0):.2f}")
                    st.text(result.get("content", "")[:200] + "...")
                    st.divider()

        # Scraped Content
        if state.get("scraped_content"):
            with st.expander(
                f"📄 Scraped Web Pages ({len(state['scraped_content'])})", expanded=False
            ):
                for i, content in enumerate(state["scraped_content"][:5], 1):
                    st.markdown(f"**{i}. {content.get('title', 'No title')}**")
                    st.text(f"URL: {content.get('url', 'N/A')}")
                    st.text(f"Words: {content.get('word_count', 0)}")
                    st.text(content.get("content", "")[:300] + "...")
                    st.divider()

    with col2:
        # ArXiv Papers
        if state.get("arxiv_papers"):
            with st.expander(f"🎓 ArXiv Papers ({len(state['arxiv_papers'])})", expanded=False):
                for i, paper in enumerate(state["arxiv_papers"], 1):
                    st.markdown(f"**{i}. {paper.get('title', 'No title')}**")
                    st.text(f"Authors: {', '.join(paper.get('authors', []))}")
                    st.text(f"Published: {paper.get('published', 'N/A')[:10]}")
                    st.text(f"URL: {paper.get('url', 'N/A')}")
                    st.text(paper.get("summary", "")[:200] + "...")
                    st.divider()

        # Retrieved Context
        if state.get("retrieved_chunks"):
            with st.expander(
                f"💡 Retrieved Context ({len(state['retrieved_chunks'])} chunks)", expanded=False
            ):
                for i, chunk in enumerate(state["retrieved_chunks"][:5], 1):
                    metadata = chunk.get("metadata", {})
                    st.markdown(
                        f"**Chunk {i}** (Similarity: {chunk.get('similarity_score', 0):.3f})"
                    )
                    st.text(f"Source: {metadata.get('title', metadata.get('url', 'Unknown'))}")
                    st.text(chunk.get("text", "")[:300] + "...")
                    st.divider()

        # Document Statistics
        if state.get("all_documents"):
            with st.expander("📊 Document Statistics", expanded=False):
                docs = state["all_documents"]
                sources: dict[str, int] = {}
                for doc in docs:
                    source = doc.get("source", "unknown")
                    sources[source] = sources.get(source, 0) + 1

                st.metric("Total Documents", len(docs))
                st.markdown("**By Source:**")
                for source, count in sources.items():
                    st.text(f"- {source}: {count}")


def run_research(topic: str) -> ResearchState | None:
    """Execute research workflow"""
    try:
        # Validate configuration
        Config.validate()

        # Create or get agent
        if st.session_state.agent is None:
            with st.spinner("Initializing research agent..."):
                st.session_state.agent = create_research_graph()

        with st.spinner("Research in progress..."):
            # Run research
            final_state: ResearchState = st.session_state.agent.research(topic)

            # Store results
            st.session_state.research_results = final_state
            st.session_state.research_history.append(
                {
                    "topic": topic,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": final_state.get("status", "unknown"),
                }
            )

            return final_state

    except Exception as e:
        st.error(f"Error during research: {str(e)}")
        st.exception(e)
        return None


def main() -> None:
    """Main application entry point"""
    initialize_session_state()
    display_header()
    display_sidebar()

    # Research input
    topic, start_button = display_research_input()

    # Execute research
    if start_button and topic:
        st.session_state.research_results = None  # Clear previous results
        final_state = run_research(topic)

        if final_state:
            st.success("✅ Research completed successfully!")
            st.balloons()

    # Display results if available
    if st.session_state.research_results:
        st.divider()
        display_research_results(st.session_state.research_results)

    # Footer
    st.divider()
    st.markdown(
        """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Autonomous Research Assistant | Powered by LangGraph, LangChain, and Groq</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
