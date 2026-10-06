"""Per-request verification values LangGraph injects into account tools.

The model never sees these arguments and cannot supply them: ToolNode fills
them from graph state, and tool_call_schema hides them from the model.
"""

from datetime import datetime
from typing import Annotated

from langgraph.prebuilt import InjectedState

VerifiedEmail = Annotated[str | None, InjectedState("verified_email")]
VerifiedAt = Annotated[datetime | None, InjectedState("verified_at")]
