from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import fastf1 as ff1
import pandas as pd

from f1sim.config import AppConfig


def setup_fastf1_cache(config: AppConfig) -> Path:
	cache_dir = Path(config.paths.fastf1_cache)
	cache_dir.mkdir(parents=True, exist_ok=True)
	ff1.Cache.enable_cache(str(cache_dir))
	return cache_dir


def load_session(season: int, round: int, session: str = "R") -> Dict[str, pd.DataFrame]:
	"""Load a FastF1 session and return key DataFrames.

	Args:
		season: Year, e.g. 2025
		round: Championship round 1-based
		session: 'FP1','FP2','FP3','Q','S','R'
	"""
	sess = ff1.get_session(season, round, session)
	sess.load()

	laps = sess.laps.copy()
	telemetry = None
	try:
		telemetry = sess.laps.get_car_data().add_distance().copy()
	except Exception:
		telemetry = pd.DataFrame()

	results = pd.DataFrame(sess.results)
	weather = sess.weather_data.copy() if getattr(sess, "weather_data", None) is not None else pd.DataFrame()
	pitstops = sess.laps[sess.laps["PitOutTime"].notna() | sess.laps["PitInTime"].notna()].copy()

	return {
		"laps": laps,
		"telemetry": telemetry,
		"results": results,
		"weather": weather,
		"pitstops": pitstops,
	} 