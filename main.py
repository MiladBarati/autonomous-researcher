"""
Main entry point for Autonomous Research Assistant

Provides CLI interface for running research tasks.
For web UI, use: streamlit run app.py
"""

import argparse
import sys

from agent.graph import create_research_graph
from agent.logger import get_logger, set_logging_context
from agent.validation import ValidationError, validate_topic
from config import Config

logger = get_logger("main")


def main() -> None:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Assistant - AI-powered research agent"
    )
    parser.add_argument("topic", nargs="?", help="Research topic to investigate")
    parser.add_argument("--web", action="store_true", help="Launch web interface (Streamlit)")

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
        # Validate and sanitize topic input
        try:
            topic = validate_topic(args.topic)
        except ValidationError as e:
            logger.error(f"Invalid topic: {e}")
            print(f"\nError: Invalid topic - {e}")
            sys.exit(1)

        # Validate configuration
        Config.validate()

        # Set logging context
        set_logging_context(topic=topic)

        logger.info("=" * 60)
        logger.info("Autonomous Research Assistant")
        logger.info("=" * 60)
        logger.info(f"\nTopic: {topic}\n")

        # Create agent and run research
        agent = create_research_graph()
        final_state = agent.research(topic)

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
        from agent.validation import sanitize_filename

        safe_topic = sanitize_filename(topic[:30].replace(" ", "_"))
        filename = f"research_report_{safe_topic}.md"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"# Research Report: {topic}\n\n")
            f.write(final_state.get("synthesis", ""))

        logger.info(f"Report saved to: {filename}")
        print(f"\nReport saved to: {filename}")
        print("=" * 60 + "\n")

    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        print(f"\nError: Invalid input - {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        print(f"\nError: Configuration issue - {e}")
        print("Please check your .env file and ensure all required API keys are set.")
        sys.exit(1)
    except (FileNotFoundError, PermissionError, OSError) as e:
        logger.error(f"File system error: {e}", exc_info=True)
        print(f"\nError: File operation failed - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Research interrupted by user")
        print("\n\nResearch interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error during research: {e}", exc_info=True)
        print(f"\nError: An unexpected error occurred - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
