from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
	data_dir: str = "data"
	cache_dir: str = "data_cache"
	models_dir: str = "models"
	fastf1_cache: str = ".fastf1"


class RuntimeConfig(BaseModel):
	random_seed: int = 42
	n_jobs: int = 4


class FastF1Config(BaseModel):
	enable_cache: bool = True
	offline: bool = False


class ModelConfig(BaseModel):
	baseline: str = "gradient_boosting"
	advanced: str = "xgboost"


class EvaluationConfig(BaseModel):
	cv_scheme: str = "season_split"
	metrics: list[str] = Field(default_factory=lambda: [
		"top3_accuracy",
		"top10_accuracy",
		"lap_mae",
		"interval_rmse",
	])


class AppConfig(BaseModel):
	paths: PathsConfig = PathsConfig()
	runtime: RuntimeConfig = RuntimeConfig()
	fastf1: FastF1Config = FastF1Config()
	model: ModelConfig = ModelConfig()
	evaluation: EvaluationConfig = EvaluationConfig()

	@staticmethod
	def load(path: str | Path = "configs/default.yaml") -> "AppConfig":
		path = Path(path)
		with path.open("r", encoding="utf-8") as f:
			data: Dict[str, Any] = yaml.safe_load(f)
		return AppConfig(**data) 