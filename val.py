#!/usr/bin/env python3
"""
Run validation for input and output structures.

This script runs validation twice:
1. For the input structure -> metrics_in
2. For the refined output structure -> metrics_out
"""

import os
import sys
import json
import click
from pathlib import Path
from CryoNetRefine.data.output.metrics_validation import run_validation


def save_metrics(metrics: dict, output_path: str) -> None:
    """Save metrics dictionary to JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    click.echo(f"Metrics saved to: {output_path}")


@click.command()
@click.argument("map_path", type=click.Path(exists=True))
@click.argument("input_pdb_path", type=click.Path(exists=True))
@click.argument("output_pdb_path", type=click.Path(exists=True))
@click.option("--resolution", type=float, required=True, help="Resolution of the density map")
@click.option("--output_dir", type=click.Path(exists=False), default="./validation_results", help="Output directory for validation results")
def validate(map_path: str, input_pdb_path: str, output_pdb_path: str, resolution: float, output_dir: str) -> None:
    """
    Run validation for both input and output structures.

    Arguments:
        map_path: Path to the density map (.mrc file)
        input_pdb_path: Path to the input structure (.pdb or .cif file)
        output_pdb_path: Path to the refined output structure (.pdb or .cif file)
        --resolution: Resolution of the density map in Angstroms
        --output_dir: Directory to save validation results
    """
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    click.echo("=" * 60)
    click.echo("Starting CryoNet.Refine Validation")
    click.echo("=" * 60)
    click.echo(f"Map path    : {map_path}")
    click.echo(f"Input PDB   : {input_pdb_path}")
    click.echo(f"Output PDB  : {output_pdb_path}")
    click.echo(f"Resolution  : {resolution} Å")
    click.echo(f"Output dir  : {output_dir}")
    click.echo("=" * 60)

    # Get base name for output files
    input_name = Path(input_pdb_path).stem
    output_name = Path(output_pdb_path).stem

    # Run validation for input structure
    click.echo(f"\n[1/2] Running validation for INPUT structure: {input_pdb_path}")
    click.echo("-" * 60)
    run_validation(map_path, input_pdb_path, resolution, "metrics_in")
    

    # Run validation for output structure
    click.echo(f"\n[2/2] Running validation for OUTPUT structure: {output_pdb_path}")
    click.echo("-" * 60)
    run_validation(map_path, output_pdb_path, resolution, "metrics_out")
    
    click.echo("\n" + "=" * 60)
    click.echo("Validation completed successfully!")
    click.echo("=" * 60)


if __name__ == "__main__":
    validate()
