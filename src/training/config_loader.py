from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_experiment_config(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data
