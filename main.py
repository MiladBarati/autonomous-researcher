"""
Main entry point for Autonomous Research Assistant

Provides CLI interface for running research tasks.
For web UI, use: streamlit run app.py
"""

import sys
import argparse
from agent.graph import create_research_graph
from agent.logger import get_logger, set_logging_context
from config import Config

logger = get_logger("main")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Assistant - AI-powered research agent"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="Research topic to investigate"
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch web interface (Streamlit)"
    )
    
    args = parser.parse_args()
    
    if args.web:
        # Launch Streamlit app
        import subprocess
        logger.info("Launching web interface...")
        subprocess.run(["streamlit", "run", "app.py"])
        return
    
    if not args.topic:
        logger.info("Autonomous Research Assistant")
        logger.info("=" * 60)
        logger.info("\nUsage:")
        logger.info("  python main.py '<research topic>'  - Run research via CLI")
        logger.info("  python main.py --web               - Launch web interface")
        logger.info("\nExample:")
        logger.info("  python main.py 'quantum computing applications'")
        logger.info("\nFor better experience, use the web interface:")
        logger.info("  streamlit run app.py")
        logger.info("=" * 60)
        sys.exit(1)
    
    try:
        # Validate configuration
        Config.validate()
        
        # Set logging context
        set_logging_context(topic=args.topic)
        
        logger.info("=" * 60)
        logger.info("Autonomous Research Assistant")
        logger.info("=" * 60)
        logger.info(f"\nTopic: {args.topic}\n")
        
        # Create agent and run research
        agent = create_research_graph()
        final_state = agent.research(args.topic)
        
        # Display results
        logger.info("=" * 60)
        logger.info("RESEARCH REPORT")
        logger.info("=" * 60)
        # Print synthesis to console for user visibility
        print("\n" + "=" * 60)
        print("RESEARCH REPORT")
        print("=" * 60 + "\n")
        print(final_state.get("synthesis", "No synthesis generated"))
        print("\n" + "=" * 60)
        
        # Save to file
        filename = f"research_report_{args.topic[:30].replace(' ', '_')}.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Research Report: {args.topic}\n\n")
            f.write(final_state.get("synthesis", ""))
        
        logger.info(f"Report saved to: {filename}")
        print(f"\nReport saved to: {filename}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Error during research: {e}", exc_info=True)
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
