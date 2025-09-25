from __future__ import annotations

from pathlib import Path

from f1sim.config import AppConfig


class PathRepository:
	def __init__(self, config: AppConfig) -> None:
		self.data_dir = Path(config.paths.data_dir)
		self.cache_dir = Path(config.paths.cache_dir)
		self.models_dir = Path(config.paths.models_dir)

	def ensure(self) -> None:
		self.data_dir.mkdir(parents=True, exist_ok=True)
		self.cache_dir.mkdir(parents=True, exist_ok=True)
		self.models_dir.mkdir(parents=True, exist_ok=True) 