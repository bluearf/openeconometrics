"""Bounded execution-order metadata without copying structured outputs."""
from __future__ import annotations

MAX_EVENT_STDOUT_BYTES = 70 * 1024
MAX_EVENT_OUTPUTS = 20


def validate_output_events(events, stdout: str, outputs: list) -> None:
    """Require one ordered reference per output and an exact stdout partition.

    Adjacent print writes share a text segment. At most one text segment can
    occur before, between or after structured displays; indices avoid repeating
    potentially large tables, models or chart data in the timeline.
    """
    if (not isinstance(events, list) or not isinstance(outputs, list)
            or len(outputs) > MAX_EVENT_OUTPUTS
            or len(events) > 2 * len(outputs) + 1
            or not isinstance(stdout, str)):
        raise ValueError("Invalid output timeline.")
    text_parts, next_index, last_type = [], 0, None
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Invalid output event.")
        kind = event.get("type")
        if kind == "stdout":
            text = event.get("text")
            if (set(event) != {"type", "text"} or not isinstance(text, str)
                    or not text or last_type == "stdout"):
                raise ValueError("Invalid output text event.")
            text_parts.append(text)
        elif kind == "output":
            index = event.get("index")
            if (set(event) != {"type", "index"} or type(index) is not int
                    or index != next_index or index >= len(outputs)):
                raise ValueError("Invalid output reference.")
            next_index += 1
        else:
            raise ValueError("Invalid output event type.")
        last_type = kind
    text = "".join(text_parts)
    if (next_index != len(outputs) or text != stdout
            or len(text.encode("utf-8")) > MAX_EVENT_STDOUT_BYTES):
        raise ValueError("Incomplete or oversized output timeline.")
