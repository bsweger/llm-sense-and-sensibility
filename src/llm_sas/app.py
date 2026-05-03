import structlog
from rich.console import Console
from rich.panel import Panel

logger = structlog.get_logger()


def main():
    """llm_sas starting point."""
    logger.info("starting llm_sas...")

    console = Console()
    console.print(
        Panel(
            ":tada: Hello from the llm_sas Python package!",
            border_style="green",
            expand=False,
            padding=(1, 4),
            subtitle="[italic]created by pyprefab[/italic]",
            subtitle_align="right",
            title="llm_sas",
            title_align="left",
        )
    )


if __name__ == "__main__":
    main()
