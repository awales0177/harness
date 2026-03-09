"""Rich-based pretty printing functions for ETL harness."""

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.layout import Layout
from rich import box
from rich.rule import Rule
from typing import TYPE_CHECKING, Any

# Avoid circular imports by using TYPE_CHECKING
if TYPE_CHECKING:
    from fluency_api import ETLContext, TableContext

# Create a global console instance
# Rich will auto-detect Jupyter notebooks and use appropriate rendering
# Removing force_terminal to allow proper Jupyter HTML rendering
console = Console(height=None)


def pretty_print_etl_context_init(context: Any) -> None:
    """Print ETL context information side by side on initialization using rich."""
    console.print()
    console.print(Rule(f"[bold blue]🚀 ETL Pipeline Initialized[/bold blue]", style="blue"))
    console.print()
    
    # Build left table (Basic Information)
    left_table = Table(show_header=False, box=None, padding=(0, 1))
    left_table.add_column(style="cyan", width=20)
    left_table.add_column(style="white")
    
    left_table.add_row("Pipeline ID", context.pipeline_id or "None")
    left_table.add_row("Source Dataset IDs", ", ".join(context.source_dataset_ids) if context.source_dataset_ids else "None")
    left_table.add_row("Periodicity", context.periodicity or "None")
    left_table.add_row("Expected Delivery", context.expected_delivery or "None")
    left_table.add_row("Delivered Version", context.delivered_version or "None")
    
    # Build right table (Organization & Product)
    right_table = Table(show_header=False, box=None, padding=(0, 1))
    right_table.add_column(style="cyan", width=20)
    right_table.add_column(style="white")
    
    right_table.add_row("ETL Platform", context.etl_platform or "None")
    right_table.add_row("GitHub Link", context.github_link or "None")
    right_table.add_row("Data Agreement ID", context.data_agreement_id or "None")
    right_table.add_row("Data Model ID", context.data_model_id or "None")
    right_table.add_row("Start Time", context.start_time.strftime("%Y-%m-%d %H:%M:%S") if context.start_time else "None")
    
    # Display side by side, each taking half the width
    console.print(Columns([left_table, right_table], equal=True, expand=True))
    console.print()


def pretty_print_source_info(table: Any) -> None:
    """Print source information for a TableContext using rich (on init when source_df is provided)."""
    console.print()
    console.print(Rule(f"[bold cyan]Initialized '{table.table_name}' Table [/bold cyan]", style="cyan"))
    console.print()
    
    # Source dimensions and data quality side by side (set at TableContext init from source_df)
    dim_panel = None
    if table.source_num_of_columns is not None or table.source_num_of_rows is not None:
        dim_table = Table(show_header=False, box=None, padding=(0, 1))
        dim_table.add_column(style="cyan", width=22)
        dim_table.add_column(style="white")
        dim_table.add_row("Column count", str(table.source_num_of_columns) if table.source_num_of_columns is not None else "—")
        dim_table.add_row("Row count", str(table.source_num_of_rows) if table.source_num_of_rows is not None else "—")
        dim_panel = Panel(dim_table, title="[bold]📐 Source Dimensions[/bold]", border_style="dim", expand=False)
    dq_panel = None
    if table.source_data_quality is not None:
        dq = table.source_data_quality
        dq_table = Table(show_header=False, box=None, padding=(0, 1))
        dq_table.add_column(style="cyan", width=22)
        dq_table.add_column(style="white")
        dq_table.add_row("Null value count", str(dq.null_value_count) if dq.null_value_count is not None else "—")
        dq_table.add_row("Empty row count", str(dq.empty_row_count) if dq.empty_row_count is not None else "—")
        dq_panel = Panel(dq_table, title="[bold]📋 Source Data Quality[/bold]", border_style="dim", expand=False)
    if dim_panel is not None and dq_panel is not None:
        console.print(Columns([dim_panel, dq_panel], equal=True, expand=True))
        console.print()
    elif dim_panel is not None:
        console.print(dim_panel)
        console.print()
    elif dq_panel is not None:
        console.print(dq_panel)
        console.print()
    
    # Build schema table
    if table.schema:
        schema_table = Table(show_header=True, box=box.SIMPLE)
        schema_table.add_column("Column Name", style="cyan")
        schema_table.add_column("Origin Type", style="yellow")
        schema_table.add_column("Data Type", style="green")
        
        for col in table.schema:
            schema_table.add_row(
                col.column_name,
                col.origin_type,
                col.data_type or ""
            )
        console.print(schema_table)
    else:
        console.print("[dim]No schema information[/dim]")
    console.print()


def pretty_print_table(table: Any, idx: int) -> None:
    """Print a single TableContext in a readable format using rich."""
    console.print()
    console.print(Rule(f"[bold green]Table {idx}: {table.table_name}[/bold green]", style="green"))
    console.print()
    
    if table.etl_comments:
        comments = table.etl_comments[:80] + "..." if len(table.etl_comments) > 80 else table.etl_comments
        console.print(f"[dim]Comments:[/dim] {comments}")
        console.print()
    
    # Functions Ran
    if table.functions_ran:
        functions_table = Table(show_header=True, box=box.SIMPLE)
        functions_table.add_column("Order", style="cyan", justify="right")
        functions_table.add_column("Function Name", style="yellow")
        functions_table.add_column("Engines", style="magenta")
        functions_table.add_column("Run ID", style="dim")
        functions_table.add_column("Duration", style="green")
        
        for func in table.functions_ran:
            engines_str = ", ".join(
                f"{eng['engine']} v{eng['version']}" for eng in func.function_engines
            ) if func.function_engines else ""
            duration = func.duration_seconds if func.duration_seconds is not None else 0.0
            duration_str = f"{duration:.4f}s" if duration > 0 else ""
            functions_table.add_row(
                str(func.run_order) if func.run_order is not None else "",
                func.function_name,
                engines_str,
                func.run_id,
                duration_str
            )
        console.print(functions_table)
        console.print()


def pretty_print_etl_context(context: Any) -> None:
    """Print a human-readable summary of the entire ETL context using rich."""
    console.print()
    console.print(Panel(
        "[bold bright_blue]ETL Context Information[/bold bright_blue]",
        border_style="bright_blue",
        expand=False
    ))
    console.print()
    
    # Basic Information Table
    console.print(Rule("[bold cyan]📊 Basic Information[/bold cyan]", style="cyan"))
    basic_table = Table(show_header=False, box=None, padding=(0, 1))
    basic_table.add_column(style="cyan", width=25)
    basic_table.add_column(style="white")
    
    basic_table.add_row("Source Dataset IDs", ", ".join(context.source_dataset_ids) if context.source_dataset_ids else "None")
    basic_table.add_row("Periodicity", context.periodicity or "None")
    basic_table.add_row("Expected Delivery", context.expected_delivery or "None")
    basic_table.add_row("Delivered Version", context.delivered_version or "None")
    console.print(basic_table)
    console.print()
    
    # ETL Organisation Table
    console.print(Rule("[bold yellow]🏢 ETL Organisation[/bold yellow]", style="yellow"))
    org_table = Table(show_header=False, box=None, padding=(0, 1))
    org_table.add_column(style="cyan", width=25)
    org_table.add_column(style="white")
    
    org_table.add_row("ETL Platform", context.etl_platform or "None")
    org_table.add_row("GitHub Link", context.github_link or "None")
    console.print(org_table)
    console.print()
    
    # Tables section
    console.print(Rule("[bold yellow]📋 Tables[/bold yellow]", style="yellow"))
    if not context.tables:
        console.print("[dim]No tables added yet.[/dim]")
    else:
        console.print(f"[bold cyan]{len(context.tables)} table(s) found[/bold cyan]")
        console.print()
        for idx, table in enumerate(context.tables, 1):
            pretty_print_table(table, idx)
    
    console.print()


def pretty_print_submit(context: Any) -> None:
    """Print submission summary using rich."""
    console.print()
    
    # Calculate duration
    duration = 0.0
    duration_str = "N/A"
    if context.start_time and context.end_time:
        duration = (context.end_time - context.start_time).total_seconds()
        if duration < 60:
            duration_str = f"{duration:.2f}s"
        elif duration < 3600:
            duration_str = f"{duration/60:.2f}m"
        else:
            duration_str = f"{duration/3600:.2f}h"
    
    # Pipeline Info card - organized into logical groups
    pipeline_info = Table(show_header=False, box=None, padding=(0, 1))
    pipeline_info.add_column(style="cyan", width=25)
    pipeline_info.add_column(style="white")
    
    # Timing
    pipeline_info.add_row("[bold]ETL Duration[/bold]", "")
    pipeline_info.add_row("  Start", str(context.start_time) if context.start_time else "N/A")
    pipeline_info.add_row("  End", str(context.end_time) if context.end_time else "N/A")
    pipeline_info.add_row("  Duration", duration_str)
    if context.etl_comments:
        pipeline_info.add_row("", "")
        pipeline_info.add_row("[bold]Comments[/bold]", "")
        comments = context.etl_comments[:45] + "..." if len(context.etl_comments) > 45 else context.etl_comments
        pipeline_info.add_row("", comments)
    
    # Display Pipeline Info card
    console.print(Panel(pipeline_info, title="[bold]Pipeline Info[/bold]", border_style="blue"))
    console.print()
    
    # Submission status
    console.print(Panel(
        "[bold green]✓[/bold green] [green]Metrics ready for submission[/green]\n"
        "[dim]Note: Actual submission to monitoring systems not yet implemented[/dim]",
        title="[bold]Submission Status[/bold]",
        border_style="green"
    ))
    console.print()
