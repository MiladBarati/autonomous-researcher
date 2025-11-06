"""
Main entry point for Autonomous Research Assistant

Provides CLI interface for running research tasks.
For web UI, use: streamlit run app.py
"""

import sys
import argparse
from agent.graph import create_research_graph
from config import Config


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
        print("Launching web interface...")
        subprocess.run(["streamlit", "run", "app.py"])
        return
    
    if not args.topic:
        print("Autonomous Research Assistant")
        print("=" * 60)
        print("\nUsage:")
        print("  python main.py '<research topic>'  - Run research via CLI")
        print("  python main.py --web               - Launch web interface")
        print("\nExample:")
        print("  python main.py 'quantum computing applications'")
        print("\nFor better experience, use the web interface:")
        print("  streamlit run app.py")
        print("=" * 60)
        sys.exit(1)
    
    try:
        # Validate configuration
        Config.validate()
        
        print("\n" + "=" * 60)
        print("Autonomous Research Assistant")
        print("=" * 60)
        print(f"\nTopic: {args.topic}\n")
        
        # Create agent and run research
        agent = create_research_graph()
        final_state = agent.research(args.topic)
        
        # Display results
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
        
        print(f"\nReport saved to: {filename}")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
