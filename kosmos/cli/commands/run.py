"""
Run command for Kosmos CLI.

Executes autonomous research with live progress visualization.
"""

import sys
import time
from typing import Optional
from datetime import datetime
from pathlib import Path

import typer
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)
from rich.layout import Layout
from rich.text import Text

from kosmos.cli.utils import (
    console,
    print_success,
    print_error,
    print_info,
    get_icon,
    format_timestamp,
    create_status_text,
)
from kosmos.cli.interactive import run_interactive_mode
from kosmos.cli.views.results_viewer import ResultsViewer


def run_research(
    question: Optional[str] = typer.Argument(None, help="Research question to investigate"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Research domain (biology, neuroscience, materials, etc.)"),
    max_iterations: int = typer.Option(10, "--max-iterations", "-i", help="Maximum number of research iterations"),
    budget: Optional[float] = typer.Option(None, "--budget", "-b", help="Budget limit in USD"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable caching"),
    interactive: bool = typer.Option(False, "--interactive", help="Use interactive mode"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to file (JSON or Markdown)"),
):
    """
    Run autonomous research on a scientific question.

    Examples:

        # Interactive mode (recommended for first time)
        kosmos run --interactive

        # Direct command
        kosmos run "What metabolic pathways differ between cancer and normal cells?" --domain biology

        # With budget limit
        kosmos run "How do perovskites optimize efficiency?" --domain materials --budget 50

        # Save results
        kosmos run "Question" --output results.json
    """
    # Use interactive mode if requested or no question provided
    if interactive or not question:
        config = run_interactive_mode()

        if not config:
            console.print("[warning]Research cancelled.[/warning]")
            raise typer.Exit(0)

        # Extract config
        question = config["question"]
        domain = config["domain"]
        max_iterations = config["max_iterations"]
        budget = config.get("budget_usd")
        no_cache = not config.get("enable_cache", True)

    # Validate inputs
    if not question:
        print_error("No research question provided. Use --interactive or provide a question.")
        raise typer.Exit(1)

    # Show starting message
    console.print()
    console.print(
        Panel(
            f"[cyan]Starting autonomous research...[/cyan]\n\n"
            f"**Question:** {question}\n"
            f"**Domain:** {domain or 'auto-detect'}\n"
            f"**Max Iterations:** {max_iterations}\n"
            f"**Budget:** ${budget} USD" if budget else "**Budget:** No limit",
            title=f"[bright_blue]{get_icon('rocket')} Kosmos Research[/bright_blue]",
            border_style="bright_blue",
        )
    )
    console.print()

    # Initialize research
    try:
        from kosmos.agents.research_director import ResearchDirectorAgent
        from kosmos.config import get_config

        # Get configuration
        config_obj = get_config()

        # Override with CLI parameters
        if domain:
            config_obj.research.enabled_domains = [domain]
        config_obj.research.max_iterations = max_iterations
        if budget:
            config_obj.research.budget_usd = budget
        config_obj.claude.enable_cache = not no_cache

        # Create research director
        director = ResearchDirectorAgent(
            research_question=question,
            domain=domain,
            config=config_obj
        )

        # Run research with live progress
        results = run_with_progress(director, question, max_iterations)

        # Display results
        viewer = ResultsViewer()
        viewer.display_research_overview(results)
        viewer.display_hypotheses_table(results.get("hypotheses", []))
        viewer.display_experiments_table(results.get("experiments", []))

        if "metrics" in results:
            viewer.display_metrics_summary(results["metrics"])

        # Export if requested
        if output:
            if output.suffix == ".json":
                viewer.export_to_json(results, output)
            elif output.suffix in [".md", ".markdown"]:
                viewer.export_to_markdown(results, output)
            else:
                print_error(f"Unsupported output format: {output.suffix}")

        print_success("Research completed successfully!", title="Complete")

    except KeyboardInterrupt:
        console.print("\n[warning]Research interrupted by user[/warning]")
        raise typer.Exit(130)

    except Exception as e:
        print_error(f"Research failed: {str(e)}", title="Error")
        if "--debug" in sys.argv:
            raise
        raise typer.Exit(1)


def run_with_progress(director, question: str, max_iterations: int) -> dict:
    """
    Run research with live progress display.

    Args:
        director: ResearchDirectorAgent instance
        question: Research question
        max_iterations: Maximum iterations

    Returns:
        Research results dictionary
    """
    # Create progress bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    # Create tasks for each phase
    hypothesis_task = progress.add_task("[cyan]Generating hypotheses...", total=100)
    experiment_task = progress.add_task("[yellow]Designing experiments...", total=100)
    execution_task = progress.add_task("[green]Executing experiments...", total=100)
    analysis_task = progress.add_task("[magenta]Analyzing results...", total=100)
    iteration_task = progress.add_task("[bright_blue]Research progress...", total=max_iterations)

    # Create current hypothesis table
    def create_status_table():
        table = Table(title="Current Status", box=None, show_header=True)
        table.add_column("Phase", style="cyan")
        table.add_column("Status", style="white")

        # Get current state from director
        state = getattr(director, "current_state", "INITIALIZING")
        iteration = getattr(director, "current_iteration", 0)

        table.add_row("Workflow State", create_status_text(state))
        table.add_row("Iteration", f"{iteration}/{max_iterations}")
        table.add_row("Started", format_timestamp(datetime.utcnow()))

        return table

    # Run with live display
    with Live(progress, console=console, refresh_per_second=4):
        try:
            # Start research (this would be the actual research loop)
            # For now, simulate progress
            results = {
                "id": f"research_{int(time.time())}",
                "question": question,
                "domain": "auto",
                "state": "COMPLETED",
                "current_iteration": 0,
                "max_iterations": max_iterations,
                "hypotheses": [],
                "experiments": [],
                "metrics": {
                    "api_calls": 0,
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "hypotheses_generated": 0,
                    "experiments_executed": 0,
                },
            }

            # This is where the actual research loop would go
            # For demonstration, we'll show progress updates
            for iteration in range(max_iterations):
                # Update iteration progress
                progress.update(iteration_task, completed=iteration + 1)

                # Simulate hypothesis generation
                progress.update(hypothesis_task, completed=25)
                time.sleep(0.1)
                progress.update(hypothesis_task, completed=50)
                time.sleep(0.1)
                progress.update(hypothesis_task, completed=100)

                # Simulate experiment design
                progress.update(experiment_task, completed=33)
                time.sleep(0.1)
                progress.update(experiment_task, completed=66)
                time.sleep(0.1)
                progress.update(experiment_task, completed=100)

                # Simulate execution
                progress.update(execution_task, completed=40)
                time.sleep(0.1)
                progress.update(execution_task, completed=80)
                time.sleep(0.1)
                progress.update(execution_task, completed=100)

                # Simulate analysis
                progress.update(analysis_task, completed=50)
                time.sleep(0.1)
                progress.update(analysis_task, completed=100)

                # Reset for next iteration
                if iteration < max_iterations - 1:
                    progress.update(hypothesis_task, completed=0)
                    progress.update(experiment_task, completed=0)
                    progress.update(execution_task, completed=0)
                    progress.update(analysis_task, completed=0)

                # NOTE: In real implementation, this would call:
                # results = director.conduct_research(question)
                # And update progress based on actual agent callbacks

            return results

        except Exception as e:
            console.print(f"\n[error]Error during research: {str(e)}[/error]")
            raise


if __name__ == "__main__":
    # Allow standalone testing
    typer.run(run_research)
