import typer
from rich import print
from pathlib import Path

import pandas as pd

from f1sim.config import AppConfig
from f1sim.storage.repository import PathRepository
from f1sim.ingest.fastf1_adapter import setup_fastf1_cache, load_session

app = typer.Typer(help="F1 Strategy Sim CLI")


@app.command()
def predict(season: int = typer.Option(..., help="F1 season year"), round: int = typer.Option(..., help="Championship round (1-based)")):
	"""Run a placeholder prediction pipeline for a given race."""
	print(f"[bold green]Predicting race[/bold green]: season={season}, round={round}")
	# TODO: Wire ingestion -> features -> model
	print("This is a scaffold. Implement pipeline in subsequent steps.")


@app.command()
def ingest(
	season: int = typer.Option(..., help="F1 season year"),
	round: int = typer.Option(..., help="Championship round (1-based)"),
	session: str = typer.Option("R", help="Session code: FP1, FP2, FP3, Q, S, R"),
	config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
):
	"""Ingest a FastF1 session and store Parquet files in data directory."""
	config = AppConfig.load(config_path)
	repo = PathRepository(config)
	repo.ensure()
	setup_fastf1_cache(config)

	print(f"[bold cyan]Loading FastF1[/bold cyan]: season={season}, round={round}, session={session}")
	dfs = load_session(season=season, round=round, session=session)

	out_dir = Path(config.paths.data_dir) / f"{season}_{round}_{session}"
	out_dir.mkdir(parents=True, exist_ok=True)

	for name, df in dfs.items():
		if isinstance(df, pd.DataFrame) and not df.empty:
			path = out_dir / f"{name}.parquet"
			# Parquet with pyarrow engine
			df.to_parquet(path, index=False)
			print(f"[green]Saved[/green] {name}: {path}")
		else:
			print(f"[yellow]Empty[/yellow] {name}, skipping save")

	print(f"[bold green]Ingestion complete[/bold green]: {out_dir}")


if __name__ == "__main__":
	app() 