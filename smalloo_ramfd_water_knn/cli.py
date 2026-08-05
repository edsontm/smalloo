from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "ramfd_water_knn_experiment.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found at {script_path}")
    sys.argv[0] = str(script_path)
    runpy.run_path(str(script_path), run_name="__main__")
