from __future__ import annotations

from pathlib import Path
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd


def _safe_seconds(td: pd.Series) -> pd.Series:
	return td.dt.total_seconds().astype(float)


def _add_penalty_features(df: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
	"""Add penalty-related features to the dataframe."""
	df = df.copy()
	
	# Grid position penalties (how many positions dropped from optimal)
	if "GridPosition" in df.columns:
		df["grid_drop"] = df["GridPosition"] - df["GridPosition"].min()
	else:
		df["grid_drop"] = 0.0
	
	# Time penalties (extract from Status or Penalty columns)
	df["time_penalty_seconds"] = 0.0
	if "Status" in results.columns:
		# Look for time penalties in status (e.g., "+5s", "+10s")
		status_penalties = results["Status"].str.extract(r'\+(\d+)s', expand=False)
		if not status_penalties.empty:
			penalty_map = pd.to_numeric(status_penalties, errors='coerce').fillna(0)
			df["time_penalty_seconds"] = df["DriverNumber"].map(
				dict(zip(results["DriverNumber"], penalty_map))
			).fillna(0)
	
	# DNF risk indicator
	df["dnf_risk"] = 0
	if "Status" in results.columns:
		dnf_statuses = results["Status"].str.contains("DNF|DNS|DSQ|NC", case=False, na=False)
		dnf_map = dnf_statuses.astype(int)
		df["dnf_risk"] = df["DriverNumber"].map(
			dict(zip(results["DriverNumber"], dnf_map))
		).fillna(0)
	
	# Points lost due to penalties (if available)
	df["points_lost"] = 0.0
	if "PointsLost" in results.columns:
		df["points_lost"] = df["DriverNumber"].map(
			dict(zip(results["DriverNumber"], results["PointsLost"]))
		).fillna(0)
	
	# Penalty severity score (combined metric)
	df["penalty_severity"] = (
		df["grid_drop"] * 0.3 +  # Grid drops are less severe
		df["time_penalty_seconds"] * 0.1 +  # Time penalties
		df["dnf_risk"] * 10.0 +  # DNF is very severe
		df["points_lost"] * 0.5  # Points lost
	)
	
	return df


def _add_pit_efficiency_features(df: pd.DataFrame, race_dir: Path) -> pd.DataFrame:
	"""Add pit stop efficiency and strategy features to the dataframe."""
	df = df.copy()
	
	# Load pit stops data
	pitstops_file = race_dir / "pitstops.parquet"
	if not pitstops_file.exists():
		# No pit stops data available
		df["avg_pit_time"] = 0.0
		df["pit_stops"] = 0
		df["pit_efficiency"] = 0.0
		df["pit_strategy_risk"] = 0.0
		return df
	
	pitstops = pd.read_parquet(pitstops_file)
	if pitstops.empty:
		df["avg_pit_time"] = 0.0
		df["pit_stops"] = 0
		df["pit_efficiency"] = 0.0
		df["pit_strategy_risk"] = 0.0
		return df
	
	# Convert pit times to seconds if they're timedelta
	for col in ["PitInTime", "PitOutTime"]:
		if col in pitstops.columns and np.issubdtype(pitstops[col].dtype, np.timedelta64):
			pitstops[col] = _safe_seconds(pitstops[col])
	
	# Calculate pit stop metrics per driver
	pit_metrics = []
	for driver in df["DriverNumber"].unique():
		driver_pits = pitstops[pitstops["DriverNumber"] == driver]
		
		if len(driver_pits) == 0:
			pit_metrics.append({
				"DriverNumber": driver,
				"avg_pit_time": 0.0,
				"pit_stops": 0,
				"pit_efficiency": 0.0,
				"pit_strategy_risk": 0.0
			})
			continue
		
		# Calculate pit stop duration (PitOutTime - PitInTime)
		if "PitInTime" in driver_pits.columns and "PitOutTime" in driver_pits.columns:
			pit_durations = driver_pits["PitOutTime"] - driver_pits["PitInTime"]
			avg_pit_time = pit_durations.mean() if len(pit_durations) > 0 else 0.0
		else:
			avg_pit_time = 0.0
		
		# Number of pit stops
		pit_stops = len(driver_pits)
		
		# Pit efficiency (lower is better, normalized by number of stops)
		pit_efficiency = avg_pit_time / (pit_stops + 1) if pit_stops > 0 else 0.0
		
		# Pit strategy risk (more stops = higher risk)
		pit_strategy_risk = pit_stops * 0.5  # Each stop adds 0.5 risk points
		
		pit_metrics.append({
			"DriverNumber": driver,
			"avg_pit_time": avg_pit_time,
			"pit_stops": pit_stops,
			"pit_efficiency": pit_efficiency,
			"pit_strategy_risk": pit_strategy_risk
		})
	
	# Convert to DataFrame and merge
	pit_df = pd.DataFrame(pit_metrics)
	df = df.merge(pit_df, on="DriverNumber", how="left")
	
	# Fill NaN values
	df["avg_pit_time"] = df["avg_pit_time"].fillna(0.0)
	df["pit_stops"] = df["pit_stops"].fillna(0)
	df["pit_efficiency"] = df["pit_efficiency"].fillna(0.0)
	df["pit_strategy_risk"] = df["pit_strategy_risk"].fillna(0.0)
	
	return df


def _add_session_pace_features(df: pd.DataFrame, race_dir: Path) -> pd.DataFrame:
	"""Add session-adjusted pace features comparing different sessions."""
	df = df.copy()
	
	# Sessions to compare (practice, qualifying vs race)
	sessions = ['fp1', 'fp2', 'fp3', 'q', 'r']
	session_paces = {}
	
	# Load pace data from each session
	for session in sessions:
		session_file = race_dir / f"{session}_laps.parquet"
		if session_file.exists():
			try:
				laps = pd.read_parquet(session_file)
				if not laps.empty and 'LapTime' in laps.columns:
					# Convert lap times to seconds
					if np.issubdtype(laps['LapTime'].dtype, np.timedelta64):
						laps['LapTime'] = _safe_seconds(laps['LapTime'])
					
					# Calculate mean pace per driver
					pace = laps.groupby('DriverNumber')['LapTime'].mean()
					session_paces[session] = pace
			except Exception:
				continue
	
	# Initialize session pace features
	df['qual_vs_race_pace'] = 0.0
	df['practice_vs_race_pace'] = 0.0
	df['pace_consistency'] = 0.0
	df['session_adaptation'] = 0.0
	
	# Qualifying vs Race pace
	if 'q' in session_paces and 'r' in session_paces:
		qual_pace = session_paces['q']
		race_pace = session_paces['r']
		
		# Calculate pace difference (race pace - qual pace)
		pace_delta = race_pace - qual_pace
		df['qual_vs_race_pace'] = df['DriverNumber'].map(pace_delta).fillna(0.0)
	
	# Practice vs Race pace (use best practice session)
	practice_sessions = [s for s in ['fp1', 'fp2', 'fp3'] if s in session_paces]
	if practice_sessions and 'r' in session_paces:
		# Use the fastest practice session for each driver
		practice_pace = pd.concat([session_paces[s] for s in practice_sessions], axis=1).min(axis=1)
		race_pace = session_paces['r']
		
		pace_delta = race_pace - practice_pace
		df['practice_vs_race_pace'] = df['DriverNumber'].map(pace_delta).fillna(0.0)
	
	# Pace consistency across sessions
	if len(session_paces) >= 2:
		# Calculate standard deviation of pace across available sessions
		all_paces = pd.concat([session_paces[s] for s in session_paces.keys()], axis=1)
		pace_std = all_paces.std(axis=1)
		df['pace_consistency'] = df['DriverNumber'].map(pace_std).fillna(0.0)
	
	# Session adaptation (how much pace improves from practice to race)
	if practice_sessions and 'r' in session_paces:
		practice_pace = pd.concat([session_paces[s] for s in practice_sessions], axis=1).mean(axis=1)
		race_pace = session_paces['r']
		
		# Positive means race pace is faster (better adaptation)
		adaptation = practice_pace - race_pace
		df['session_adaptation'] = df['DriverNumber'].map(adaptation).fillna(0.0)
	
	return df


def _add_team_driver_form_features(df: pd.DataFrame, race_dir: Path, 
                                  data_root: Path) -> pd.DataFrame:
	"""Add team and driver performance trends from historical data."""
	df = df.copy()
	
	# Initialize form features
	df['team_form_trend'] = 0.0
	df['driver_form_trend'] = 0.0
	df['team_consistency'] = 0.0
	df['driver_consistency'] = 0.0
	
	# Get historical race data
	historical_races = _get_historical_races(data_root, race_dir)
	if not historical_races:
		return df
	
	# Team form trends (last 3 races)
	team_trends = _calculate_team_trends(historical_races)
	if team_trends:
		df['team_form_trend'] = df['TeamName'].map(team_trends).fillna(0.0)
	
	# Driver form trends (last 5 races)
	driver_trends = _calculate_driver_trends(historical_races)
	if driver_trends:
		df['driver_form_trend'] = df['DriverNumber'].map(driver_trends).fillna(0.0)
	
	# Team consistency (position variance)
	team_consistency = _calculate_team_consistency(historical_races)
	if team_consistency:
		df['team_consistency'] = df['TeamName'].map(team_consistency).fillna(0.0)
	
	# Driver consistency (position variance)
	driver_consistency = _calculate_driver_consistency(historical_races)
	if driver_consistency:
		df['driver_consistency'] = df['DriverNumber'].map(driver_consistency).fillna(0.0)
	
	return df


def _get_historical_races(data_root: Path, current_race_dir: Path) -> list[pd.DataFrame]:
	"""Get historical race results for form analysis."""
	historical_races = []
	
	# Extract current race info
	race_name = current_race_dir.name
	parts = race_name.split('_')
	if len(parts) >= 2:
		current_season = int(parts[0])
		current_round = int(parts[1])
		
		# Look for races from the same season (earlier rounds)
		for race_dir in data_root.glob(f"{current_season}_*_R"):
			race_parts = race_dir.name.split('_')
			if len(race_parts) >= 2:
				race_round = int(race_parts[1])
				if race_round < current_round:  # Only earlier races
					results_file = race_dir / "results.parquet"
					if results_file.exists():
						try:
							results = pd.read_parquet(results_file)
							if not results.empty and 'Position' in results.columns:
								historical_races.append(results)
						except Exception:
							continue
	
	return historical_races


def _calculate_team_trends(historical_races: list[pd.DataFrame]) -> dict[str, float]:
	"""Calculate team performance trends (improving/declining)."""
	team_trends = {}
	
	# Group results by team
	team_results = {}
	for race in historical_races:
		if 'TeamName' in race.columns and 'Position' in race.columns:
			for _, row in race.iterrows():
				team = row['TeamName']
				position = row['Position']
				if pd.notna(team) and pd.notna(position):
					if team not in team_results:
						team_results[team] = []
					team_results[team].append(position)
	
	# Calculate trends for each team
	for team, positions in team_results.items():
		if len(positions) >= 2:
			# Simple linear trend (negative = improving, positive = declining)
			x = np.arange(len(positions))
			y = positions
			try:
				slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
				team_trends[team] = -slope  # Negative slope = improving
			except:
				team_trends[team] = 0.0
	
	return team_trends


def _calculate_driver_trends(historical_races: list[pd.DataFrame]) -> dict[str, float]:
	"""Calculate driver performance trends (improving/declining)."""
	driver_trends = {}
	
	# Group results by driver
	driver_results = {}
	for race in historical_races:
		if 'DriverNumber' in race.columns and 'Position' in race.columns:
			for _, row in race.iterrows():
				driver = str(row['DriverNumber'])
				position = row['Position']
				if pd.notna(driver) and pd.notna(position):
					if driver not in driver_results:
						driver_results[driver] = []
					driver_results[driver].append(position)
	
	# Calculate trends for each driver
	for driver, positions in driver_results.items():
		if len(positions) >= 2:
			# Simple linear trend (negative = improving, positive = declining)
			x = np.arange(len(positions))
			y = positions
			try:
				slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
				driver_trends[driver] = -slope  # Negative slope = improving
			except:
				driver_trends[driver] = 0.0
	
	return driver_trends


def _calculate_team_consistency(historical_races: list[pd.DataFrame]) -> dict[str, float]:
	"""Calculate team consistency (lower variance = more consistent)."""
	team_consistency = {}
	
	# Group results by team
	team_results = {}
	for race in historical_races:
		if 'TeamName' in race.columns and 'Position' in race.columns:
			for _, row in race.iterrows():
				team = row['TeamName']
				position = row['Position']
				if pd.notna(team) and pd.notna(position):
					if team not in team_results:
						team_results[team] = []
					team_results[team].append(position)
	
	# Calculate consistency for each team
	for team, positions in team_results.items():
		if len(positions) >= 2:
			# Lower variance = more consistent (better)
			variance = np.var(positions)
			team_consistency[team] = -variance  # Negative variance = more consistent
		else:
			team_consistency[team] = 0.0
	
	return team_consistency


def _calculate_driver_consistency(historical_races: list[pd.DataFrame]) -> dict[str, float]:
	"""Calculate driver consistency (lower variance = more consistent)."""
	driver_consistency = {}
	
	# Group results by driver
	driver_results = {}
	for race in historical_races:
		if 'DriverNumber' in race.columns and 'Position' in race.columns:
			for _, row in race.iterrows():
				driver = str(row['DriverNumber'])
				position = row['Position']
				if pd.notna(driver) and pd.notna(position):
					if driver not in driver_results:
						driver_results[driver] = []
					driver_results[driver].append(position)
	
	# Calculate consistency for each driver
	for driver, positions in driver_results.items():
		if len(positions) >= 2:
			# Lower variance = more consistent (better)
			variance = np.var(positions)
			driver_consistency[driver] = -variance  # Negative variance = more consistent
		else:
			driver_consistency[driver] = 0.0
	
	return driver_consistency


def build_features_from_dir(race_dir: str | Path) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
	"""Build simple per-driver features and targets from a race data directory.

	Returns:
		X: features per driver
		y: target finishing position (1=win)
		meta: metadata including driver, team, grid, season, round
	"""
	race_dir = Path(race_dir)
	laps_pq = race_dir / "laps.parquet"
	results_pq = race_dir / "results.parquet"
	weather_pq = race_dir / "weather.parquet"

	if not laps_pq.exists() or not results_pq.exists():
		raise FileNotFoundError(f"Missing required parquet in {race_dir}")

	laps = pd.read_parquet(laps_pq)
	results = pd.read_parquet(results_pq)
	weather = pd.read_parquet(weather_pq) if weather_pq.exists() else pd.DataFrame()

	# Basic lap features per driver
	# Convert LapTime, Sector times to seconds where available
	for c in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
		if c in laps and np.issubdtype(laps[c].dtype, np.timedelta64):
			laps[c] = _safe_seconds(laps[c])

	lap_agg = (
		laps.groupby("DriverNumber").agg(
			mean_lap_time=("LapTime", "mean"),
			best_lap_time=("LapTime", "min"),
			std_lap_time=("LapTime", "std"),
			n_laps=("LapNumber", "count"),
			n_pit_laps=("PitOutTime", lambda s: s.notna().sum()),
		)
		.reset_index()
	)

	# Results carry target and metadata
	meta_cols = [
		"DriverNumber",
		"Abbreviation",
		"TeamName",
		"GridPosition",
		"Position",
	]
	meta = results[[c for c in meta_cols if c in results.columns]].copy()

	# Merge
	df = meta.merge(lap_agg, on="DriverNumber", how="left")

	# Weather features (session means)
	if not weather.empty:
		w_cols = [c for c in ["AirTemp", "TrackTemp", "Humidity", "WindSpeed"] if c in weather.columns]
		if w_cols:
			w_mean = weather[w_cols].mean().to_frame().T
			# Broadcast same session means to each driver
			for c in w_cols:
				df[c] = float(w_mean.iloc[0][c])

	# Penalty features
	df = _add_penalty_features(df, results)
	
	# Pit efficiency features
	df = _add_pit_efficiency_features(df, race_dir)
	
	# Session pace features
	df = _add_session_pace_features(df, race_dir)
	
	# Team/driver form features
	df = _add_team_driver_form_features(df, race_dir, race_dir.parent)

	# Replace inf/nan with comprehensive cleaning
	X = df.copy()
	
	# Fill NaN values with sensible defaults
	X = X.fillna({
		"std_lap_time": 0.0, 
		"mean_lap_time": X["mean_lap_time"].median(), 
		"best_lap_time": X["best_lap_time"].median(), 
		"n_pit_laps": 0, 
		"n_laps": 0,
		# Weather features
		"AirTemp": 25.0,
		"TrackTemp": 35.0, 
		"Humidity": 50.0,
		"WindSpeed": 5.0,
		# Penalty features
		"grid_drop": 0.0,
		"time_penalty_seconds": 0.0,
		"dnf_risk": 0.0,
		"points_lost": 0.0,
		"penalty_severity": 0.0,
		# Pit features
		"avg_pit_time": 0.0,
		"pit_stops": 0,
		"pit_efficiency": 0.0,
		"pit_strategy_risk": 0.0,
		# Session pace features
		"qual_vs_race_pace": 0.0,
		"practice_vs_race_pace": 0.0,
		"pace_consistency": 0.0,
		"session_adaptation": 0.0,
		# Form features
		"team_form_trend": 0.0,
		"driver_form_trend": 0.0,
		"team_consistency": 0.0,
		"driver_consistency": 0.0,
		# Advanced features
		"race_pace_delta": 0.0,
		"driver_form_meanlap_prev": X["mean_lap_time"].median()
	})
	
	# Replace infinite values with finite ones
	X = X.replace([np.inf, -np.inf], np.nan)
	X = X.fillna(0.0)

	# Target: final classified position as integer; fallback to large number if NaN
	y = X["Position"].fillna(99).astype(int)

	# Feature selection (exclude identifiers and target)
	drop_cols = ["Position", "Abbreviation", "TeamName", "DriverNumber"]
	feature_cols = [c for c in X.columns if c not in drop_cols]

	meta_out = X[["DriverNumber", "Abbreviation", "TeamName", "GridPosition"]].copy() if "Abbreviation" in X.columns else X[["DriverNumber", "GridPosition"]].copy()
	X_out = X[feature_cols].copy()

	return X_out, y, meta_out


def parse_race_id(race_id: str) -> tuple[int, int, str]:
	"""Parse race id like '2024-1' or '2024-1-R' into (season, round, session)."""
	parts = race_id.split("-")
	season = int(parts[0])
	round_ = int(parts[1])
	session = parts[2] if len(parts) > 2 else "R"
	return season, round_, session


def build_features_for_races(data_root: str | Path, race_ids: List[str]) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
	"""Build concatenated features/targets/meta for multiple races.

	Returns X_all, y_all, meta_all, groups (race id per row) for CV.
	"""
	root = Path(data_root)
	frames_X: list[pd.DataFrame] = []
	frames_y: list[pd.Series] = []
	frames_meta: list[pd.DataFrame] = []
	groups: list[str] = []
	# Simple form store: prior mean lap time per driver across processed races
	prior_mean_by_driver: dict[str, float] = {}
	for rid in race_ids:
		season, round_, session = parse_race_id(rid)
		race_dir = root / f"{season}_{round_}_{session}"
		X, y, meta = build_features_from_dir(race_dir)
		# Normalize types
		meta["DriverNumber"] = meta["DriverNumber"].astype(str)
		# Race pace delta: mean lap time minus race median
		race_median = float(X["mean_lap_time"].median()) if "mean_lap_time" in X else 0.0
		X["race_pace_delta"] = X.get("mean_lap_time", 0.0) - race_median
		# Simple driver form feature: prior mean of mean_lap_time across previous races
		form_vals = []
		for dn, mlt in zip(meta["DriverNumber"].tolist(), X.get("mean_lap_time", pd.Series([np.nan]*len(X))).tolist()):
			prev = prior_mean_by_driver.get(dn)
			form_vals.append(prev if prev is not None else race_median)
		X["driver_form_meanlap_prev"] = form_vals
		# Update prior store
		for dn, mlt in zip(meta["DriverNumber"].tolist(), X.get("mean_lap_time", pd.Series([np.nan]*len(X))).tolist()):
			if pd.notna(mlt):
				prior_mean_by_driver[dn] = float(mlt) if dn not in prior_mean_by_driver else (prior_mean_by_driver[dn] * 0.5 + float(mlt) * 0.5)

		frames_X.append(X)
		frames_y.append(y)
		meta2 = meta.copy()
		rid_std = f"{season}-{round_}-{session}"
		meta2["race_id"] = rid_std
		frames_meta.append(meta2)
		groups.extend([rid_std] * len(X))

	X_all = pd.concat(frames_X, axis=0, ignore_index=True)
	y_all = pd.concat(frames_y, axis=0, ignore_index=True)
	meta_all = pd.concat(frames_meta, axis=0, ignore_index=True)
	groups_s = pd.Series(groups, name="race_id")
	return X_all, y_all, meta_all, groups_s 