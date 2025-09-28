from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error


class PositionRegressor:
	def __init__(self) -> None:
		self.model = GradientBoostingRegressor(random_state=42)

	def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
		self.model.fit(X, y)

	def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
		if hasattr(self.model, "feature_names_in_"):
			required = list(self.model.feature_names_in_)
			Xc = X.copy()
			# Add missing with sensible defaults
			for col in required:
				if col not in Xc.columns:
					Xc[col] = 0.0
			# Extra columns are ignored by selecting required order
			Xc = Xc[required]
			return Xc
		return X

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		X_aligned = self._align_features(X)
		return self.model.predict(X_aligned)

	def save(self, path: str | Path) -> None:
		Path(path).parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(self.model, path)

	@staticmethod
	def load(path: str | Path) -> "PositionRegressor":
		mdl = PositionRegressor()
		mdl.model = joblib.load(path)
		return mdl


def cross_val_mae(X: pd.DataFrame, y: pd.Series, groups: pd.Series | None = None, n_splits: int = 5) -> float:
	gkf = GroupKFold(n_splits=n_splits) if groups is not None else None
	maes: list[float] = []
	if gkf is None:
		for _ in range(n_splits):
			mdl = PositionRegressor()
			mdl.fit(X, y)
			y_pred = mdl.predict(X)
			maes.append(mean_absolute_error(y, y_pred))
	else:
		for tr_idx, te_idx in gkf.split(X, y, groups):
			mdl = PositionRegressor()
			mdl.fit(X.iloc[tr_idx], y.iloc[tr_idx])
			y_pred = mdl.predict(X.iloc[te_idx])
			maes.append(mean_absolute_error(y.iloc[te_idx], y_pred))
	return float(np.mean(maes))


def predict_order(pred_positions: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
	rank = pd.Series(pred_positions).rank(method="first").astype(int)
	out = meta.copy()
	out["pred_position"] = pred_positions
	out["pred_rank"] = rank
	out = out.sort_values("pred_position")
	out["pred_final_pos"] = np.arange(1, len(out) + 1)
	return out


def estimate_finish_gaps(order_df: pd.DataFrame, X: pd.DataFrame, 
                        race_dir: Path | None = None) -> pd.DataFrame:
	"""Enhanced gap estimation with tire degradation and stint analysis.

	If X lacks 'race_pace_delta', compute it as mean_lap_time - median(mean_lap_time).
	"""
	res = order_df.copy()
	# Align X rows to order_df by DriverNumber (from order_df, not X)
	key = "DriverNumber"
	X2 = X.copy()
	# Add DriverNumber from order_df for alignment
	X2[key] = order_df[key].astype(str)
	
	# Base race pace delta
	if "race_pace_delta" not in X2.columns:
		if "mean_lap_time" in X2.columns:
			med = float(pd.to_numeric(X2["mean_lap_time"], errors="coerce").median())
			X2["race_pace_delta"] = pd.to_numeric(X2["mean_lap_time"], errors="coerce") - med
		else:
			X2["race_pace_delta"] = 0.0
	
	# Add tire degradation factor if lap data is available
	if race_dir and (race_dir / "laps.parquet").exists():
		try:
			laps = pd.read_parquet(race_dir / "laps.parquet")
			tire_degradation = _calculate_tire_degradation(laps)
			X2 = X2.merge(tire_degradation, on=key, how="left")
			X2["tire_degradation"] = X2["tire_degradation"].fillna(0.0)
		except Exception:
			X2["tire_degradation"] = 0.0
	else:
		X2["tire_degradation"] = 0.0
	
	# Add pit strategy impact
	if "pit_strategy_risk" in X2.columns:
		pit_impact = X2["pit_strategy_risk"] * 0.05  # Each pit stop adds 0.05s risk
	else:
		pit_impact = pd.Series([0.0] * len(X2))
	
	res[key] = res[key].astype(str)
	merge_cols = [key, "race_pace_delta", "tire_degradation"]
	res = res.merge(X2[merge_cols], on=key, how="left")
	
	# Enhanced gap calculation
	base_gap = (
		pd.to_numeric(res["race_pace_delta"], errors="coerce").fillna(0.0) +
		pd.to_numeric(res["tire_degradation"], errors="coerce").fillna(0.0) * 0.1 +
		pit_impact.values
	)
	
	# Normalize and accumulate
	base_gap = base_gap - np.nanmin(base_gap)
	scale = 0.15  # seconds per unit delta (reduced from 0.2 for more realistic gaps)
	
	# Order gaps according to predicted finishing order
	res = res.sort_values("pred_final_pos").reset_index(drop=True)
	gaps = np.cumsum(base_gap[res.index]) * scale
	res["est_gap_to_winner_s"] = gaps
	
	return res


def _calculate_tire_degradation(laps: pd.DataFrame) -> pd.DataFrame:
	"""Calculate tire degradation rate for each driver."""
	if laps.empty or 'DriverNumber' not in laps.columns or 'LapTime' not in laps.columns:
		return pd.DataFrame(columns=['DriverNumber', 'tire_degradation'])
	
	# Convert lap times to seconds if needed
	if np.issubdtype(laps['LapTime'].dtype, np.timedelta64):
		laps = laps.copy()
		laps['LapTime'] = laps['LapTime'].dt.total_seconds()
	
	degradation_data = []
	for driver in laps['DriverNumber'].unique():
		driver_laps = laps[laps['DriverNumber'] == driver].sort_values('LapNumber')
		
		if len(driver_laps) < 5:  # Need at least 5 laps for meaningful degradation
			degradation_data.append({'DriverNumber': driver, 'tire_degradation': 0.0})
			continue
		
		# Calculate pace degradation over the race
		lap_times = pd.to_numeric(driver_laps['LapTime'], errors='coerce').dropna()
		if len(lap_times) < 3:
			degradation_data.append({'DriverNumber': driver, 'tire_degradation': 0.0})
			continue
		
		# Simple linear degradation rate (seconds per lap)
		x = np.arange(len(lap_times))
		y = lap_times.values
		try:
			slope = np.polyfit(x, y, 1)[0] if len(x) > 1 else 0
			degradation_rate = abs(slope)  # Positive degradation rate
		except:
			degradation_rate = 0.0
		
		degradation_data.append({
			'DriverNumber': driver, 
			'tire_degradation': degradation_rate
		})
	
	return pd.DataFrame(degradation_data) 