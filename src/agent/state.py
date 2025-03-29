"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class State:
    """Defines the input state for the agent, representing a narrower interface to the outside world.

    This class is used to define the initial state and structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    for more information.
    """

    user_input: str = ""  # The user's question
    csv_path: Optional[Path] = None  # Path to the CSV file
    csv_preview: Optional[str] = None  # Preview of the CSV data
    generated_code: Optional[str] = None  # The generated Python code
    csv_metadata: Dict = field(default_factory=dict)  # Metadata about the CSV (columns, types, etc.)
    error_message: Optional[str] = None  # Any error messages
