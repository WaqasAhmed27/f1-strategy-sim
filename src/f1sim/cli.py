import typer
from rich import print
from pathlib import Path

import pandas as pd

from f1sim.config import AppConfig
from f1sim.storage.repository import PathRepository
from f1sim.ingest.fastf1_adapter import setup_fastf1_cache, load_session
from f1sim.features.builder import build_features_from_dir, build_features_for_races, parse_race_id
from f1sim.models.baseline import PositionRegressor, predict_order, estimate_finish_gaps

app = typer.Typer(help="F1 Strategy Sim CLI")


def ensure_data_for_race(data_root: str, season: int, race_round: int, session: str, config: AppConfig | None = None) -> Path:
	root = Path(data_root)
	out_dir = root / f"{season}_{race_round}_{session}"
	required = [out_dir / "laps.parquet", out_dir / "results.parquet"]
	if all(p.exists() for p in required):
		return out_dir
	# Auto-ingest if missing
	if config is None:
		config = AppConfig.load("configs/default.yaml")
	PathRepository(config).ensure()
	setup_fastf1_cache(config)
	dfs = load_session(season=season, round=race_round, session=session)
	out_dir.mkdir(parents=True, exist_ok=True)
	for name, df in dfs.items():
		if isinstance(df, pd.DataFrame) and not df.empty:
			(df.to_parquet(out_dir / f"{name}.parquet", index=False, engine="pyarrow"))
	return out_dir


@app.command()
def predict(
	season: int = typer.Option(..., help="F1 season year"),
	race_round: int = typer.Option(..., help="Championship round (1-based)"),
	session: str = typer.Option("R", help="Session code; uses ingested directory"),
	model_path: str = typer.Option("models/baseline_position.joblib", help="Path to saved model"),
	data_root: str = typer.Option("data", help="Data directory root"),
):
	"""Predict finishing order using a trained model and ingested data, with simple gap estimates."""
	print(f"[bold green]Predicting race[/bold green]: season={season}, round={race_round}")
	race_dir = Path(data_root) / f"{season}_{race_round}_{session}"
	X, y, meta = build_features_from_dir(race_dir)
	mdl = PositionRegressor.load(model_path)
	pred = mdl.predict(X)
	order_df = predict_order(pred, meta)
	order_df = estimate_finish_gaps(order_df, X, race_dir)
	cols = ["pred_final_pos", "Abbreviation", "TeamName", "pred_position", "est_gap_to_winner_s"] if "Abbreviation" in order_df.columns else ["pred_final_pos", "DriverNumber", "pred_position", "est_gap_to_winner_s"]
	print(order_df[cols])


@app.command()
def ingest(
	season: int = typer.Option(..., help="F1 season year"),
	race_round: int = typer.Option(..., help="Championship round (1-based)"),
	session: str = typer.Option("R", help="Session code: FP1, FP2, FP3, Q, S, R"),
	config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config"),
):
	"""Ingest a FastF1 session and store Parquet files in data directory."""
	config = AppConfig.load(config_path)
	repo = PathRepository(config)
	repo.ensure()
	setup_fastf1_cache(config)

	print(f"[bold cyan]Loading FastF1[/bold cyan]: season={season}, round={race_round}, session={session}")
	dfs = load_session(season=season, round=race_round, session=session)

	out_dir = Path(config.paths.data_dir) / f"{season}_{race_round}_{session}"
	out_dir.mkdir(parents=True, exist_ok=True)

	for name, df in dfs.items():
		if isinstance(df, pd.DataFrame) and not df.empty:
			path = out_dir / f"{name}.parquet"
			# Parquet with pyarrow engine
			df.to_parquet(path, index=False, engine="pyarrow")
			print(f"[green]Saved[/green] {name}: {path}")
		else:
			print(f"[yellow]Empty[/yellow] {name}, skipping save")

	print(f"[bold green]Ingestion complete[/bold green]: {out_dir}")


@app.command()
def train(
	season: int = typer.Option(..., help="F1 season year"),
	race_round: int = typer.Option(..., help="Championship round (1-based)"),
	session: str = typer.Option("R", help="Session code; uses ingested directory"),
	model_path: str = typer.Option("models/baseline_position.joblib", help="Where to save the model"),
):
	"""Train baseline model on a single race (demo)."""
	race_dir = Path("data") / f"{season}_{race_round}_{session}"
	X, y, meta = build_features_from_dir(race_dir)
	mdl = PositionRegressor()
	mdl.fit(X, y)
	mdl.save(model_path)
	print(f"[bold green]Model saved[/bold green]: {model_path}")


@app.command(name="predict-race")
def predict_race(
	season: int = typer.Option(..., help="F1 season year"),
	race_round: int = typer.Option(..., help="Championship round (1-based)"),
	session: str = typer.Option("R", help="Session code; uses ingested directory"),
	model_path: str = typer.Option("models/baseline_position.joblib", help="Path to saved model"),
):
	"""Alias of predict for backwards compatibility."""
	return predict(season=season, race_round=race_round, session=session, model_path=model_path)


@app.command()
def eval(
	season: int = typer.Option(..., help="F1 season year"),
	race_round: int = typer.Option(..., help="Championship round (1-based)"),
	session: str = typer.Option("R", help="Session code; uses ingested directory"),
	model_path: str = typer.Option("models/baseline_position.joblib", help="Path to saved model"),
):
	"""Evaluate predictions vs. actual results for a single race (Top3/Top10/MAE)."""
	race_dir = Path("data") / f"{season}_{race_round}_{session}"
	X, y_true, meta = build_features_from_dir(race_dir)
	results_pq = race_dir / "results.parquet"
	true_df = pd.read_parquet(results_pq)
	true_pos = true_df[["DriverNumber", "Position"]].copy()

	mdl = PositionRegressor.load(model_path)
	pred = mdl.predict(X)
	pred_df = predict_order(pred, meta)
	pred_df = pred_df.merge(true_pos, on="DriverNumber", how="left")
	pred_df.rename(columns={"Position": "true_position"}, inplace=True)

	# Position MAE on final classified positions
	pos_mae = float((pred_df["pred_final_pos"] - pred_df["true_position"]).abs().mean())

	# Top-N accuracy by set overlap
	pred_top3 = set(pred_df.nsmallest(3, "pred_final_pos")["DriverNumber"].tolist())
	true_top3 = set(true_df.nsmallest(3, "Position")["DriverNumber"].tolist())
	top3_acc = len(pred_top3 & true_top3) / 3.0

	pred_top10 = set(pred_df.nsmallest(10, "pred_final_pos")["DriverNumber"].tolist())
	true_top10 = set(true_df.nsmallest(10, "Position")["DriverNumber"].tolist())
	top10_acc = len(pred_top10 & true_top10) / 10.0

	print({"pos_mae": round(pos_mae, 3), "top3_acc": round(top3_acc, 3), "top10_acc": round(top10_acc, 3)})


@app.command()
def train_multi(
	races: str = typer.Option(..., help="Comma-separated race ids like '2024-1,2024-2' (session defaults to R)"),
	model_path: str = typer.Option("models/baseline_position.joblib", help="Where to save the model"),
	data_root: str = typer.Option("data", help="Data directory root"),
	config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config (for auto-ingest)"),
	cv_splits: int = typer.Option(3, help="Group K-Fold splits (by race_id)"),
):
	"""Train baseline model on multiple races with group CV reporting."""
	race_ids = [r.strip() for r in races.split(",") if r.strip()]
	config = AppConfig.load(config_path)
	for rid in race_ids:
		season, rnd, session = parse_race_id(rid)
		ensure_data_for_race(data_root, season, rnd, session, config)
	X, y, meta, groups = build_features_for_races(data_root, race_ids)
	# Group CV MAE by race
	from sklearn.model_selection import GroupKFold
	from sklearn.metrics import mean_absolute_error

	gkf = GroupKFold(n_splits=min(cv_splits, len(set(groups))))
	maes = []
	for tr, te in gkf.split(X, y, groups):
		mdl = PositionRegressor()
		mdl.fit(X.iloc[tr], y.iloc[tr])
		yp = mdl.predict(X.iloc[te])
		maes.append(mean_absolute_error(y.iloc[te], yp))
	cv_mae = float(pd.Series(maes).mean()) if maes else float("nan")

	mdl = PositionRegressor()
	mdl.fit(X, y)
	mdl.save(model_path)
	print({"samples": len(X), "races": int(pd.Series(groups).nunique()), "cv_mae": round(cv_mae, 3), "model_path": model_path})


@app.command()
def eval_multi(
	races: str = typer.Option(..., help="Comma-separated race ids like '2024-1,2024-2' (session defaults to R)"),
	model_path: str = typer.Option("models/baseline_position.joblib", help="Path to saved model"),
	data_root: str = typer.Option("data", help="Data directory root"),
	config_path: str = typer.Option("configs/default.yaml", help="Path to YAML config (for auto-ingest)"),
):
	"""Evaluate on multiple races (report averaged metrics)."""
	race_ids = [r.strip() for r in races.split(",") if r.strip()]
	config = AppConfig.load(config_path)
	for rid in race_ids:
		season, rnd, session = parse_race_id(rid)
		ensure_data_for_race(data_root, season, rnd, session, config)
	X, y_true, meta, groups = build_features_for_races(data_root, race_ids)
	mdl = PositionRegressor.load(model_path)
	pred = mdl.predict(X)
	pred_df = predict_order(pred, meta)
	pred_df["race_id"] = groups.values
	pred_df["DriverNumber"] = pred_df["DriverNumber"].astype(str)

	# Merge truth
	truth_rows = []
	for rid in race_ids:
		season, rnd, session = parse_race_id(rid)
		race_dir = Path(data_root) / f"{season}_{rnd}_{session}"
		true_df_all = pd.read_parquet(race_dir / "results.parquet")
		true_df_all["DriverNumber"] = true_df_all["DriverNumber"].astype(str)
		true_df = true_df_all[["DriverNumber", "Position"]].copy()
		true_df["race_id"] = f"{season}-{rnd}-{session}"
		truth_rows.append(true_df)
	truth = pd.concat(truth_rows, ignore_index=True)

	joined = pred_df.merge(truth, on=["race_id", "DriverNumber"], how="left")
	joined.rename(columns={"Position": "true_position"}, inplace=True)

	# Metrics per race
	metrics = []
	for rid, grp in joined.groupby("race_id"):
		grp2 = grp.dropna(subset=["true_position"])  # drop rows without truth
		pos_mae = float((grp2["pred_final_pos"] - grp2["true_position"]).abs().mean()) if not grp2.empty else float("nan")
		pred_top3 = set(grp2.nsmallest(3, "pred_final_pos")["DriverNumber"].tolist()) if len(grp2) >= 3 else set()
		true_top3 = set(grp2.nsmallest(3, "true_position")["DriverNumber"].tolist()) if len(grp2) >= 3 else set()
		top3_acc = len(pred_top3 & true_top3) / 3.0 if pred_top3 and true_top3 else float("nan")
		pred_top10 = set(grp2.nsmallest(10, "pred_final_pos")["DriverNumber"].tolist()) if len(grp2) >= 10 else set()
		true_top10 = set(grp2.nsmallest(10, "true_position")["DriverNumber"].tolist()) if len(grp2) >= 10 else set()
		top10_acc = len(pred_top10 & true_top10) / 10.0 if pred_top10 and true_top10 else float("nan")
		metrics.append({"race_id": rid, "pos_mae": pos_mae, "top3_acc": top3_acc, "top10_acc": top10_acc})

	m = pd.DataFrame(metrics)
	print(m)
	print({
		"races": int(m["race_id"].nunique()),
		"pos_mae": round(float(m["pos_mae"].dropna().mean()), 3) if m["pos_mae"].notna().any() else float("nan"),
		"top3_acc": round(float(m["top3_acc"].dropna().mean()), 3) if m["top3_acc"].notna().any() else float("nan"),
		"top10_acc": round(float(m["top10_acc"].dropna().mean()), 3) if m["top10_acc"].notna().any() else float("nan"),
	})


if __name__ == "__main__":
	app() 