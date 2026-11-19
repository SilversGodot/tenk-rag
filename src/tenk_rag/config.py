from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class PipelineSettings:
    chunk_strategy: str = "section"
    chunk_size: int = 400
    chunk_overlap: int = 50
    retrieval: str = "hybrid"
    top_k: int = 5
    candidates: int = 30
    rerank: bool = False
    use_filters: bool = True

    @classmethod
    def from_config(cls, cfg: dict, experiment: str | None = None) -> "PipelineSettings":
        values = dict(cfg.get("defaults", {}))
        if experiment:
            exps = cfg.get("experiments", {})
            if experiment not in exps:
                raise KeyError(f"Unknown experiment '{experiment}'. Options: {list(exps)}")
            values.update(exps[experiment] or {})
        known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def index_name(self) -> str:
        """Indexes depend only on chunking, so experiments that change only
        retrieval settings share one index."""
        return f"{self.chunk_strategy}_{self.chunk_size}_{self.chunk_overlap}"

    def to_dict(self) -> dict:
        return asdict(self)
