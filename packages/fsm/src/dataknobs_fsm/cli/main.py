"""FSM CLI tool for managing and executing FSM configurations.

This module provides a command-line interface for:
- Creating and validating FSM configurations
- Running FSM executions with data
- Managing FSM history and checkpoints
- Debugging and profiling FSM operations
"""

import click
import json
from dataknobs_fsm.utils.json_encoder import dumps as json_dumps
import yaml
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
import asyncio

from .. import __version__
from ..api.simple import SimpleFSM
from ..api.advanced import AdvancedFSM
from ..config.loader import ConfigLoader
from ..execution.history import ExecutionHistory
from ..storage.file import FileStorage
from ..patterns.etl import create_etl_pipeline, ETLMode
from ..patterns.file_processing import create_csv_processor, create_json_stream_processor

console = Console()


@click.group()
@click.version_option(version=__version__)
def cli():
    """FSM CLI - Finite State Machine Management Tool"""
    pass


@cli.group()
def config():
    """FSM configuration management commands"""
    pass


@config.command()
@click.argument('template', type=click.Choice(['basic', 'etl', 'workflow', 'processing']))
@click.option('--output', '-o', default='fsm_config.yaml', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml')
def create(template: str, output: str, format: str):
    """Create a new FSM configuration from template"""
    templates = {
        'basic': {
            'name': 'Basic_FSM',
            'data_mode': 'copy',
            'states': [
                {'name': 'start', 'is_start': True},
                {'name': 'process'},
                {'name': 'end', 'is_end': True}
            ],
            'arcs': [
                {'from': 'start', 'to': 'process', 'name': 'begin'},
                {'from': 'process', 'to': 'end', 'name': 'complete'}
            ]
        },
        'etl': {
            'name': 'ETL_Pipeline',
            'data_mode': 'copy',
            'resources': [
                {'name': 'source_db', 'type': 'database', 'provider': 'sqlite', 'path': 'source.db'},
                {'name': 'target_db', 'type': 'database', 'provider': 'sqlite', 'path': 'target.db'}
            ],
            'states': [
                {'name': 'extract', 'is_start': True, 'resources': ['source_db']},
                {'name': 'transform'},
                {'name': 'load', 'resources': ['target_db']},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {'from': 'extract', 'to': 'transform', 'name': 'extracted'},
                {'from': 'transform', 'to': 'load', 'name': 'transformed'},
                {'from': 'load', 'to': 'complete', 'name': 'loaded'}
            ]
        },
        'workflow': {
            'name': 'Workflow_FSM',
            'data_mode': 'reference',
            'states': [
                {'name': 'receive', 'is_start': True},
                {'name': 'validate'},
                {'name': 'approve'},
                {'name': 'reject'},
                {'name': 'complete', 'is_end': True}
            ],
            'arcs': [
                {'from': 'receive', 'to': 'validate', 'name': 'received'},
                {'from': 'validate', 'to': 'approve', 'name': 'valid'},
                {'from': 'validate', 'to': 'reject', 'name': 'invalid'},
                {'from': 'approve', 'to': 'complete', 'name': 'approved'},
                {'from': 'reject', 'to': 'complete', 'name': 'rejected'}
            ]
        },
        'processing': {
            'name': 'File_Processor',
            'data_mode': 'direct',
            'states': [
                {'name': 'read', 'is_start': True},
                {'name': 'parse'},
                {'name': 'process'},
                {'name': 'write'},
                {'name': 'done', 'is_end': True}
            ],
            'arcs': [
                {'from': 'read', 'to': 'parse', 'name': 'file_read'},
                {'from': 'parse', 'to': 'process', 'name': 'parsed'},
                {'from': 'process', 'to': 'write', 'name': 'processed'},
                {'from': 'write', 'to': 'done', 'name': 'written'}
            ]
        }
    }
    
    config_data = templates[template]
    
    output_path = Path(output)
    if format == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
    else:
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    console.print(f"[green][/green] Created {template} configuration in {output}")
    console.print("\nConfiguration overview:")
    console.print(f"  Name: {config_data['name']}")
    console.print(f"  States: {len(config_data['states'])}")
    console.print(f"  Arcs: {len(config_data['arcs'])}")
    console.print(f"  Data Mode: {config_data['data_mode']}")


@config.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--verbose', '-v', is_flag=True, help='Show detailed validation output')
def validate(config_file: str, verbose: bool):
    """Validate an FSM configuration file"""
    loader = ConfigLoader()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading configuration...", total=None)
            
            config = loader.load_from_file(config_file)
            progress.update(task, description="Validating configuration...")
            
            # Configuration is already validated by loading it
            is_valid = True
            errors = []
            
            progress.stop()
        
        if is_valid:
            console.print("[green][/green] Configuration is valid!")
            
            if verbose:
                console.print("\n[bold]Configuration Details:[/bold]")
                console.print(f"  Name: {config.name}")
                console.print(f"  Data Mode: {config.data_mode}")
                
                # Count states and arcs across all networks
                total_states = sum(len(net.states) for net in config.networks)
                total_arcs = 0
                for net in config.networks:
                    for state in net.states:
                        if hasattr(state, 'arcs') and state.arcs:
                            total_arcs += len(state.arcs)
                
                console.print(f"  States: {total_states}")
                console.print(f"  Arcs: {total_arcs}")
                
                if config.resources:
                    console.print(f"  Resources: {len(config.resources)}")
        else:
            console.print("[red][/red] Configuration validation failed!")
            console.print("\n[bold red]Errors:[/bold red]")
            for error in errors:
                console.print(f"  {error}")
            sys.exit(1)
            
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@config.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--format', '-f', type=click.Choice(['tree', 'graph', 'table']), default='tree')
def show(config_file: str, format: str):
    """Display FSM configuration structure"""
    loader = ConfigLoader()
    
    try:
        config = loader.load_from_file(config_file)
        
        if format == 'tree':
            tree = Tree(f"[bold]{config.name}[/bold]")
            
            # Show networks
            for network in config.networks:
                network_branch = tree.add(f"Network: {network.name}")
                
                states_branch = network_branch.add("States")
                for state in network.states:
                    state_label = state.name
                    if state.is_start:
                        state_label += " [green](start)[/green]"
                    if state.is_end:
                        state_label += " [red](end)[/red]"
                    states_branch.add(state_label)
                
                arcs_branch = network_branch.add("Arcs")
                for state in network.states:
                    if hasattr(state, 'arcs') and state.arcs:
                        for arc in state.arcs:
                            arc_label = f"{state.name} → {arc.target}"
                            if hasattr(arc, 'name') and arc.name:
                                arc_label += f" [{arc.name}]"
                            arcs_branch.add(arc_label)
            
            if config.resources:
                resources_branch = tree.add("Resources")
                for resource in config.resources:
                    resources_branch.add(f"{resource.name}: {resource.type}")
            
            console.print(tree)
            
        elif format == 'table':
            # States table
            states_table = Table(title=f"{config.name} - States")
            states_table.add_column("Network", style="blue")
            states_table.add_column("Name", style="cyan")
            states_table.add_column("Type", style="green")
            
            for network in config.networks:
                for state in network.states:
                    state_type = []
                    if state.is_start:
                        state_type.append("Start")
                    if state.is_end:
                        state_type.append("End")
                    if not state_type:
                        state_type.append("Normal")
                    
                    states_table.add_row(
                        network.name,
                        state.name,
                        ' '.join(state_type)
                    )
            
            console.print(states_table)
            
            # Arcs table
            arcs_table = Table(title=f"{config.name} - Arcs")
            arcs_table.add_column("Network", style="blue")
            arcs_table.add_column("From", style="cyan")
            arcs_table.add_column("To", style="cyan")
            arcs_table.add_column("Name", style="yellow")
            
            for network in config.networks:
                for state in network.states:
                    if hasattr(state, 'arcs') and state.arcs:
                        for arc in state.arcs:
                            arc_name = arc.name if hasattr(arc, 'name') and arc.name else '-'
                            arcs_table.add_row(
                                network.name,
                                state.name,
                                arc.target,
                                arc_name
                            )
            
            console.print(arcs_table)
            
        elif format == 'graph':
            console.print("[yellow]Graph visualization (Mermaid format)[/yellow]")
            console.print("\n```mermaid")
            console.print("graph TD")
            
            # Process all networks
            for network in config.networks:
                if len(config.networks) > 1:
                    console.print(f"    subgraph {network.name}")
                
                for state in network.states:
                    shape = "([{}])" if state.is_start else \
                            "(({}))" if state.is_end else \
                            "[{}]"
                    console.print(f"    {state.name}{shape.format(state.name)}")
                
                # Collect arcs from states
                for state in network.states:
                    if hasattr(state, 'arcs') and state.arcs:
                        for arc in state.arcs:
                            arc_label = arc.name if hasattr(arc, 'name') and arc.name else "transition"
                            console.print(f"    {state.name} -->|{arc_label}| {arc.target}")
                
                if len(config.networks) > 1:
                    console.print("    end")
            
            console.print("```")
            
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        sys.exit(1)


@cli.group()
def run():
    """Execute FSM operations"""
    pass


@run.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--data', '-d', help='Input data (JSON string or file path)')
@click.option('--initial-state', '-s', help='Initial state name')
@click.option('--timeout', '-t', type=float, help='Execution timeout in seconds')
@click.option('--output', '-o', help='Output file for results')
@click.option('--verbose', '-v', is_flag=True, help='Show execution details')
def execute(config_file: str, data: str | None, initial_state: str | None,
           timeout: float | None, output: str | None, verbose: bool):
    """Execute FSM with data"""
    # Parse input data
    input_data = {}
    if data:
        if Path(data).exists():
            with open(data) as f:
                input_data = json.load(f)
        else:
            try:
                input_data = json.loads(data)
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON data[/red]")
                sys.exit(1)
    
    # Create and run FSM
    fsm = SimpleFSM(config_file)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        progress.add_task("Executing FSM...", total=None)
        
        try:
            result = fsm.process(
                data=input_data,
                initial_state=initial_state,
                timeout=timeout
            )
            progress.stop()
            
            if result.get('success', False):
                console.print("[green][/green] Execution completed successfully!")
                console.print(f"  Final state: {result.get('final_state', 'unknown')}")
                path = result.get('path', [])
                console.print(f"  Transitions: {len(path) - 1 if path else 0}")
                
                if verbose:
                    console.print("\n[bold]Execution Path:[/bold]")
                    for i, state in enumerate(result.get('path', [])):
                        console.print(f"  {i+1}. {state}")
                    
                    if 'data' in result:
                        console.print("\n[bold]Final Data:[/bold]")
                        console.print(Syntax(
                            json_dumps(result['data'], indent=2),
                            "json",
                            theme="monokai"
                        ))
                
                if output:
                    with open(output, 'w') as f:
                        json.dump(result, f, indent=2)
                    console.print(f"\n[green]Results saved to {output}[/green]")
                    
            else:
                console.print("[red][/red] Execution failed!")
                console.print(f"  Error: {result.get('error', 'Unknown error')}")
                sys.exit(1)
                
        except Exception as e:
            progress.stop()
            console.print(f"[red]Execution error: {e}[/red]")
            sys.exit(1)


@run.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--batch-size', '-b', type=int, default=10, help='Batch size')
@click.option('--workers', '-w', type=int, default=4, help='Number of parallel workers')
@click.option('--output', '-o', help='Output file for results')
@click.option('--progress', '-p', is_flag=True, help='Show progress bar')
def batch(config_file: str, data_file: str, batch_size: int, workers: int,
         output: str | None, progress: bool):
    """Execute FSM on batch data"""
    # Load batch data
    with open(data_file) as f:
        if data_file.endswith('.jsonl'):
            batch_data = [json.loads(line) for line in f]
        else:
            batch_data = json.load(f)
            if not isinstance(batch_data, list):
                console.print("[red]Data file must contain a JSON array or JSONL[/red]")
                sys.exit(1)
    
    # Create FSM
    fsm = SimpleFSM(config_file)
    
    console.print(f"Processing {len(batch_data)} items...")
    console.print(f"  Batch size: {batch_size}")
    console.print(f"  Workers: {workers}")
    
    try:
        if progress:
            with Progress(console=console) as prog:
                task = prog.add_task("Processing...", total=len(batch_data))
                
                results = []
                for i in range(0, len(batch_data), batch_size):
                    batch = batch_data[i:i+batch_size]
                    batch_results = fsm.process_batch(
                        data=batch,
                        batch_size=batch_size,
                        max_workers=workers
                    )
                    results.extend(batch_results)
                    prog.update(task, advance=len(batch))
        else:
            results = fsm.process_batch(
                data=batch_data,
                batch_size=batch_size,
                max_workers=workers
            )
        
        # Calculate statistics
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        console.print("\n[bold]Results:[/bold]")
        console.print(f"  Total: {len(results)}")
        console.print(f"  [green]Successful: {successful}[/green]")
        if failed > 0:
            console.print(f"  [red]Failed: {failed}[/red]")
        
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"\n[green]Results saved to {output}[/green]")
            
    except Exception as e:
        console.print(f"[red]Batch processing error: {e}[/red]")
        sys.exit(1)


@run.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.argument('source')
@click.option('--sink', '-s', help='Output sink (file path or URL)')
@click.option('--chunk-size', '-c', type=int, default=100, help='Stream chunk size')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json')
def stream(config_file: str, source: str, sink: str | None, 
          chunk_size: int, format: str):
    """Process streaming data through FSM"""
    # Create FSM
    fsm = SimpleFSM(config_file)
    
    console.print("Starting stream processing...")
    console.print(f"  Source: {source}")
    if sink:
        console.print(f"  Sink: {sink}")
    console.print(f"  Chunk size: {chunk_size}")
    
    async def run_stream():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Processing stream...", total=None)
            
            try:
                result = await fsm.process_stream(
                    source=source,
                    sink=sink,
                    chunk_size=chunk_size
                )
                progress.stop()
                
                console.print("\n[green][/green] Stream processing completed!")
                console.print(f"  Records processed: {result.get('total_processed', 0)}")
                console.print(f"  Chunks: {result.get('chunks_processed', 0)}")
                if 'errors' in result and result['errors'] > 0:
                    console.print(f"  [yellow]Errors: {result['errors']}[/yellow]")
                    
            except Exception as e:
                progress.stop()
                console.print(f"[red]Stream processing error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run_stream())


@cli.group()
def debug():
    """Debug and profile FSM operations"""
    pass


@debug.command()  # type: ignore
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--data', '-d', help='Input data (JSON string or file path)')
@click.option('--breakpoint', '-b', multiple=True, help='Set breakpoint at state')
@click.option('--trace', '-t', is_flag=True, help='Enable execution tracing')
@click.option('--profile', '-p', is_flag=True, help='Enable performance profiling')
def run(config_file: str, data: str | None, breakpoint: tuple, 
       trace: bool, profile: bool):
    """Debug FSM execution with breakpoints and tracing"""
    # Load configuration
    loader = ConfigLoader()
    config = loader.load_from_file(config_file)
    
    # Parse input data
    input_data = {}
    if data:
        if Path(data).exists():
            with open(data) as f:
                input_data = json.load(f)
        else:
            try:
                input_data = json.loads(data)
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON data[/red]")
                sys.exit(1)
    
    # Create advanced FSM
    fsm = AdvancedFSM(config)
    
    # Set breakpoints
    for bp in breakpoint:
        fsm.set_breakpoint(bp)
        console.print(f"[yellow]Breakpoint set at state: {bp}[/yellow]")
    
    async def run_debug():
        try:
            if trace:
                console.print("[cyan]Tracing enabled[/cyan]\n")
                trace_log = await fsm.trace_execution(input_data)
                
                # Display trace
                table = Table(title="Execution Trace")
                table.add_column("Time", style="cyan")
                table.add_column("State", style="green")
                table.add_column("Arc", style="yellow")
                table.add_column("Event")
                
                for entry in trace_log:
                    table.add_row(
                        entry['timestamp'],
                        entry.get('state', '-'),
                        entry.get('arc', '-'),
                        entry['event']
                    )
                
                console.print(table)
                
            elif profile:
                console.print("[cyan]Profiling enabled[/cyan]\n")
                profile_data = await fsm.profile_execution(input_data)
                
                # Display profile
                console.print("[bold]Performance Profile:[/bold]")
                console.print(f"  Total time: {profile_data['total_time']:.3f}s")
                console.print(f"  Transitions: {profile_data['transition_count']}")
                
                if 'state_times' in profile_data:
                    console.print("\n[bold]State Execution Times:[/bold]")
                    for state, time in profile_data['state_times'].items():
                        console.print(f"  {state}: {time:.3f}s")
                
                if 'arc_times' in profile_data:
                    console.print("\n[bold]Arc Transition Times:[/bold]")
                    for arc, time in profile_data['arc_times'].items():
                        console.print(f"  {arc}: {time:.3f}s")
                        
            else:
                # Interactive debugging
                from ..api.advanced import FSMDebugger
                
                debugger = FSMDebugger(fsm, config)
                await debugger.start_session(input_data)
                
        except Exception as e:
            console.print(f"[red]Debug error: {e}[/red]")
            sys.exit(1)
    
    asyncio.run(run_debug())


@cli.group()
def history():
    """Manage FSM execution history"""
    pass


@history.command(name='list')  # Explicitly set command name
@click.option('--fsm-name', '-n', help='Filter by FSM name')
@click.option('--limit', '-l', type=int, default=10, help='Number of entries to show')
@click.option('--format', '-f', type=click.Choice(['table', 'json']), default='table')
def list_history(fsm_name: str | None, limit: int, format: str):
    """List execution history"""
    # Create history manager with file backend
    storage = FileStorage(Path.home() / '.fsm' / 'history')
    manager = ExecutionHistory(storage)  # type: ignore
    
    # Query history
    entries = asyncio.run(manager.query_history(
        fsm_name=fsm_name,
        limit=limit
    ))
    
    if not entries:
        console.print("[yellow]No history entries found[/yellow]")
        return
    
    if format == 'table':
        table = Table(title="Execution History")
        table.add_column("Execution ID", style="cyan")
        table.add_column("FSM Name", style="green")
        table.add_column("Start Time")
        table.add_column("End Time")
        table.add_column("Status", style="yellow")
        table.add_column("States")
        
        for entry in entries:
            status = "Success" if entry.get('success') else "Failed"
            states = len(entry.get('states', []))
            
            table.add_row(
                entry['execution_id'][:8],
                entry['fsm_name'],
                entry['start_time'],
                entry.get('end_time', '-'),
                status,
                str(states)
            )
        
        console.print(table)
    else:
        console.print(json_dumps(entries, indent=2))


@history.command()
@click.argument('execution_id')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def show_execution(execution_id: str, verbose: bool):
    """Show details of a specific execution"""
    # Create history manager
    storage = FileStorage(Path.home() / '.fsm' / 'history')
    manager = ExecutionHistory(storage)  # type: ignore
    
    # Get execution details
    entry = asyncio.run(manager.get_execution(execution_id))
    
    if not entry:
        console.print(f"[red]Execution {execution_id} not found[/red]")
        sys.exit(1)
    
    console.print("[bold]Execution Details:[/bold]")
    console.print(f"  ID: {entry['execution_id']}")
    console.print(f"  FSM: {entry['fsm_name']}")
    console.print(f"  Start: {entry['start_time']}")
    console.print(f"  End: {entry.get('end_time', 'In progress')}")
    console.print(f"  Status: {'Success' if entry.get('success') else 'Failed'}")
    
    if verbose:
        console.print("\n[bold]Execution Path:[/bold]")
        for i, state in enumerate(entry.get('states', [])):
            console.print(f"  {i+1}. {state['name']} @ {state['timestamp']}")
        
        if 'arcs' in entry:
            console.print("\n[bold]Transitions:[/bold]")
            for arc in entry['arcs']:
                console.print(f"  {arc['from']} � {arc['to']} [{arc['name']}]")
        
        if 'error' in entry:
            console.print("\n[bold red]Error:[/bold red]")
            console.print(f"  {entry['error']}")


@cli.group()
def pattern():
    """Run pre-configured FSM patterns"""
    pass


@pattern.command()
@click.option('--source', '-s', required=True, help='Source database connection')
@click.option('--target', '-t', required=True, help='Target database connection')
@click.option('--mode', '-m', type=click.Choice(['full', 'incremental', 'upsert']), 
              default='full')
@click.option('--batch-size', '-b', type=int, default=1000)
@click.option('--checkpoint', '-c', help='Resume from checkpoint ID')
def etl(source: str, target: str, mode: str, batch_size: int, checkpoint: str | None):
    """Run ETL pipeline pattern"""
    console.print("[bold]Starting ETL Pipeline[/bold]")
    console.print(f"  Source: {source}")
    console.print(f"  Target: {target}")
    console.print(f"  Mode: {mode}")
    console.print(f"  Batch size: {batch_size}")
    
    if checkpoint:
        console.print(f"  Resuming from checkpoint: {checkpoint}")
    
    # Create ETL pipeline
    etl_mode = ETLMode[mode.upper()]
    pipeline = create_etl_pipeline(
        source=source,
        target=target,
        mode=etl_mode,
        batch_size=batch_size
    )
    
    async def run_etl():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Running ETL...", total=None)
            
            try:
                metrics = await pipeline.run(checkpoint_id=checkpoint)
                progress.stop()
                
                console.print("\n[green][/green] ETL completed successfully!")
                console.print(f"  Records extracted: {metrics['extracted']}")
                console.print(f"  Records loaded: {metrics['loaded']}")
                if metrics['errors'] > 0:
                    console.print(f"  [yellow]Errors: {metrics['errors']}[/yellow]")
                    
            except Exception as e:
                progress.stop()
                console.print(f"[red]ETL error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run_etl())


@pattern.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['csv', 'json']), default='csv')
@click.option('--transform', '-t', multiple=True, help='Transformation functions')
@click.option('--filter', '-F', multiple=True, help='Filter expressions')
def process_file(input_file: str, output: str | None, format: str,
                transform: tuple, filter: tuple):
    """Process file using FSM pattern"""
    console.print(f"[bold]Processing {format.upper()} file[/bold]")
    console.print(f"  Input: {input_file}")
    if output:
        console.print(f"  Output: {output}")
    
    # Create processor based on format
    if format == 'csv':
        processor = create_csv_processor(
            input_file=input_file,
            output_file=output
        )
    else:
        processor = create_json_stream_processor(
            input_file=input_file,
            output_file=output
        )
    
    async def run_processing():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            progress.add_task("Processing file...", total=None)
            
            try:
                metrics = await processor.process()
                progress.stop()
                
                console.print("\n[green][/green] File processing completed!")
                console.print(f"  Lines read: {metrics.get('lines_read', 0)}")
                console.print(f"  Records processed: {metrics['records_processed']}")
                console.print(f"  Records written: {metrics.get('records_written', 0)}")
                if metrics.get('errors', 0) > 0:
                    console.print(f"  [yellow]Errors: {metrics['errors']}[/yellow]")
                    
            except Exception as e:
                progress.stop()
                console.print(f"[red]Processing error: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run_processing())


def main():
    """Main entry point for CLI"""
    cli()


if __name__ == '__main__':
    main()
