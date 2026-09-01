"""server-sent events formatting, shared by every streaming route"""

import json
from typing import Any


def sse_event(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
