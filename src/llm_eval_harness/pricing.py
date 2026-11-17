"""
pricing.py

Per-token pricing table used to turn token usage into cost.

Prices are configuration, not code: nothing here is a real vendor price list.
Supply your own YAML/JSON (see ``examples/pricing.yaml``) and keep it in sync
with your provider's published rates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelPrice(BaseModel):
    """USD price per one million tokens."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_per_million: float = Field(..., ge=0.0)
    completion_per_million: float = Field(..., ge=0.0)


class PricingTable(BaseModel):
    """Model-name -> price mapping with an optional fallback price."""

    model_config = ConfigDict(extra="forbid")

    models: dict[str, ModelPrice] = Field(default_factory=dict)
    default: ModelPrice | None = Field(
        default=None, description="Used for models not listed. If unset, unknown models raise."
    )

    def price_for(self, model: str) -> ModelPrice:
        if model in self.models:
            return self.models[model]
        if self.default is not None:
            return self.default
        raise KeyError(
            f"No price configured for model {model!r} and no default price set. "
            f"Known models: {sorted(self.models)}"
        )

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost in USD of one call."""
        price = self.price_for(model)
        return (
            prompt_tokens * price.prompt_per_million
            + completion_tokens * price.completion_per_million
        ) / 1_000_000


def load_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML or JSON file whose top level must be a mapping."""
    text = path.read_text(encoding="utf-8")
    data = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level.")  # noqa: TRY004
    return data


def load_pricing(path: str | Path) -> PricingTable:
    """Load a pricing table from a .yaml/.yml/.json file."""
    return PricingTable.model_validate(load_mapping(Path(path)))
