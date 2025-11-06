"""
LangGraph Workflow for Autonomous Research Agent

Defines the state graph with nodes for:
- Research planning
- Web search
- Content scraping
- ArXiv search
- Document processing
- Vector storage
- Retrieval and synthesis
"""

from typing import Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langgraph.graph import StateGraph, END

from agent.state import ResearchState
from agent.tools import ToolManager
from agent.rag import RAGPipeline
from config import get_llm, Config


class ResearchAgent:
    """Autonomous Research Agent using LangGraph"""
    
    def __init__(self):
        self.llm = get_llm()
        self.tools = ToolManager()
        self.rag = RAGPipeline()
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state machine"""
        
        # Create state graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("plan_research", self.plan_research)
        workflow.add_node("search_web", self.search_web)
        workflow.add_node("scrape_content", self.scrape_content)
        workflow.add_node("search_arxiv", self.search_arxiv)
        workflow.add_node("process_documents", self.process_documents)
        workflow.add_node("embed_and_store", self.embed_and_store)
        workflow.add_node("retrieve_and_synthesize", self.retrieve_and_synthesize)
        
        # Set entry point
        workflow.set_entry_point("plan_research")
        
        # Add edges
        workflow.add_edge("plan_research", "search_web")
        workflow.add_edge("search_web", "scrape_content")
        workflow.add_edge("scrape_content", "search_arxiv")
        workflow.add_edge("search_arxiv", "process_documents")
        workflow.add_edge("process_documents", "embed_and_store")
        workflow.add_edge("embed_and_store", "retrieve_and_synthesize")
        workflow.add_edge("retrieve_and_synthesize", END)
        
        return workflow.compile()
    
    def plan_research(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node: Create research plan and generate search queries.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with research plan and queries
        """
        print("\n=== PLANNING RESEARCH ===")
        
        topic = state["topic"]
        
        # Create planning prompt
        planning_prompt = f"""You are an autonomous research agent. Your task is to create a comprehensive research plan for the following topic:

Topic: {topic}

Please:
1. Analyze the topic and identify key areas to research
2. Generate 3-5 specific search queries that will help gather comprehensive information
3. Explain your research strategy

Format your response as:
RESEARCH PLAN:
[Your analysis and strategy]

SEARCH QUERIES:
1. [Query 1]
2. [Query 2]
3. [Query 3]
...
"""
        
        # Get LLM response
        messages = [SystemMessage(content="You are a research planning expert."),
                   HumanMessage(content=planning_prompt)]
        
        response = self.llm.invoke(messages)
        plan_text = response.content
        
        # Extract search queries from response
        queries = []
        lines = plan_text.split('\n')
        in_queries_section = False
        
        for line in lines:
            line = line.strip()
            if 'SEARCH QUERIES:' in line:
                in_queries_section = True
                continue
            if in_queries_section and line and (line[0].isdigit() or line.startswith('-')):
                # Extract query text, removing numbering
                query = line.split('.', 1)[-1].strip() if '.' in line else line.lstrip('- ')
                if query:
                    queries.append(query)
        
        # Fallback: if no queries extracted, use the topic
        if not queries:
            queries = [topic, f"{topic} overview", f"{topic} recent developments"]
        
        print(f"Research Plan Created")
        print(f"Generated {len(queries)} search queries")
        
        return {
            "research_plan": plan_text,
            "search_queries": queries,
            "messages": state["messages"] + [HumanMessage(content=planning_prompt), 
                                            AIMessage(content=plan_text)],
            "step_count": state["step_count"] + 1,
            "status": "planned"
        }
    
    def search_web(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node: Execute web searches using Tavily.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with search results
        """
        print("\n=== SEARCHING WEB ===")
        
        queries = state["search_queries"]
        all_results = []
        
        for i, query in enumerate(queries[:5], 1):  # Limit to 5 queries
            print(f"Query {i}/{len(queries[:5])}: {query}")
            results = self.tools.tavily.search(query, max_results=3)
            all_results.extend(results)
            print(f"  Found {len(results)} results")
        
        print(f"Total search results: {len(all_results)}")
        
        return {
            "search_results": all_results,
            "step_count": state["step_count"] + 1,
            "status": "searched"
        }
    
    def scrape_content(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node: Scrape full content from search result URLs.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with scraped content
        """
        print("\n=== SCRAPING CONTENT ===")
        
        search_results = state["search_results"]
        urls = [result["url"] for result in search_results if result.get("url")]
        
        print(f"Scraping {min(len(urls), Config.MAX_SCRAPE_URLS)} URLs...")
        scraped = self.tools.scraper.scrape_multiple(urls, max_urls=Config.MAX_SCRAPE_URLS)
        
        print(f"Successfully scraped {len(scraped)} pages")
        
        return {
            "scraped_content": scraped,
            "step_count": state["step_count"] + 1,
            "status": "scraped"
        }
    
    def search_arxiv(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node: Search ArXiv for academic papers.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with arxiv papers
        """
        print("\n=== SEARCHING ARXIV ===")
        
        topic = state["topic"]
        papers = self.tools.arxiv.search(topic, max_results=Config.MAX_ARXIV_RESULTS)
        
        print(f"Found {len(papers)} ArXiv papers")
        
        return {
            "arxiv_papers": papers,
            "step_count": state["step_count"] + 1,
            "status": "arxiv_searched"
        }
    
    def process_documents(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node: Process PDFs and combine all documents.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with processed documents
        """
        print("\n=== PROCESSING DOCUMENTS ===")
        
        all_docs = []
        
        # Add scraped web content
        all_docs.extend(state["scraped_content"])
        print(f"Added {len(state['scraped_content'])} web documents")
        
        # Add arxiv papers (summaries)
        for paper in state["arxiv_papers"]:
            all_docs.append({
                "title": paper["title"],
                "content": f"{paper['title']}\n\nAuthors: {', '.join(paper['authors'])}\n\n{paper['summary']}",
                "url": paper["url"],
                "source": "arxiv",
                "authors": paper["authors"]
            })
        print(f"Added {len(state['arxiv_papers'])} ArXiv paper summaries")
        
        # Optional: Process ArXiv PDFs (commented out for speed, can be enabled)
        # pdf_content = self.tools.pdf_processor.extract_from_arxiv_papers(state["arxiv_papers"][:2])
        # all_docs.extend(pdf_content)
        # print(f"Processed {len(pdf_content)} PDFs")
        
        print(f"Total documents: {len(all_docs)}")
        
        return {
            "all_documents": all_docs,
            "step_count": state["step_count"] + 1,
            "status": "documents_processed"
        }
    
    def embed_and_store(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node: Embed documents and store in vector database.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with vector store info
        """
        print("\n=== EMBEDDING AND STORING ===")
        
        documents = state["all_documents"]
        
        if not documents:
            print("No documents to store")
            return {
                "step_count": state["step_count"] + 1,
                "status": "no_documents"
            }
        
        # Store documents in vector DB
        chunk_count = self.rag.store_documents(documents)
        
        stats = self.rag.get_collection_stats()
        print(f"Vector store stats: {stats}")
        
        return {
            "vector_store_id": self.rag.collection_name,
            "step_count": state["step_count"] + 1,
            "status": "stored"
        }
    
    def retrieve_and_synthesize(self, state: ResearchState) -> Dict[str, Any]:
        """
        Node: Retrieve relevant context and synthesize final report.
        
        Args:
            state: Current research state
            
        Returns:
            Updated state with synthesis
        """
        print("\n=== RETRIEVING AND SYNTHESIZING ===")
        
        topic = state["topic"]
        
        # Retrieve relevant chunks
        print("Retrieving relevant context...")
        retrieved_chunks = self.rag.retrieve(topic, top_k=10)
        print(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # Format context
        context = self.rag.format_retrieved_context(retrieved_chunks)
        
        # Create synthesis prompt
        synthesis_prompt = f"""You are a research analyst tasked with synthesizing information on the following topic:

Topic: {topic}

Based on the research gathered (see context below), create a comprehensive research report that:
1. Provides a clear overview of the topic
2. Summarizes key findings and insights
3. Highlights important facts, statistics, and developments
4. Cites sources appropriately
5. Is well-structured with sections

CONTEXT FROM RESEARCH:
{context}

Please write a comprehensive, well-organized research report (aim for 800-1500 words):
"""
        
        # Generate synthesis
        print("Generating synthesis...")
        messages = [
            SystemMessage(content="You are an expert research analyst who synthesizes information into clear, comprehensive reports."),
            HumanMessage(content=synthesis_prompt)
        ]
        
        response = self.llm.invoke(messages)
        synthesis = response.content
        
        print(f"Synthesis completed ({len(synthesis)} characters)")
        
        return {
            "retrieved_chunks": retrieved_chunks,
            "synthesis": synthesis,
            "messages": state["messages"] + [HumanMessage(content=synthesis_prompt),
                                            AIMessage(content=synthesis)],
            "step_count": state["step_count"] + 1,
            "status": "completed"
        }
    
    def research(self, topic: str) -> Dict[str, Any]:
        """
        Execute full research workflow.
        
        Args:
            topic: Research topic
            
        Returns:
            Final state with synthesis
        """
        from agent.state import create_initial_state
        
        print(f"\n{'='*60}")
        print(f"STARTING AUTONOMOUS RESEARCH")
        print(f"Topic: {topic}")
        print(f"{'='*60}")
        
        # Create initial state
        initial_state = create_initial_state(topic)
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"RESEARCH COMPLETED")
        print(f"Status: {final_state['status']}")
        print(f"Steps taken: {final_state['step_count']}")
        print(f"{'='*60}\n")
        
        return final_state


def create_research_graph() -> ResearchAgent:
    """
    Factory function to create research agent.
    
    Returns:
        Configured ResearchAgent instance
    """
    return ResearchAgent()

