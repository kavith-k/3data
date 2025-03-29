"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional, Dict, Any

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    # Model configuration
    model_name: str = "anthropic/claude-3.7-sonnet"  # Default model
    model_temperature: float = 0.0  # Low temperature for more deterministic responses
    
    # CSV processing settings
    csv_preview_rows: int = 5  # Number of rows to include in preview
    max_csv_size_mb: int = 10  # Maximum CSV file size in MB
    
    # Code generation settings
    include_pandas: bool = True  # Whether to include pandas in generated code
    include_matplotlib: bool = True  # Whether to include matplotlib in generated code
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
