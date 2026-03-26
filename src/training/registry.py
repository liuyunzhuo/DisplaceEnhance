from __future__ import annotations

from typing import Any, Callable, Dict, Optional


class Registry:
    def __init__(self, name: str) -> None:
        self.name = name
        self._items: Dict[str, Any] = {}

    def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
        def _wrap(obj: Any) -> Any:
            key = name or obj.__name__
            if key in self._items:
                raise KeyError(f"{key} already registered in {self.name}")
            self._items[key] = obj
            return obj

        return _wrap

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"{name} not found in {self.name}")
        return self._items[name]


MODEL_REGISTRY = Registry("model")
DATASET_REGISTRY = Registry("dataset")
NETWORK_REGISTRY = Registry("network")
