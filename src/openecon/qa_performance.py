"""Opt-in synthetic native QA recorder. Never attached by a normal desktop build."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    phase: Literal["shell", "project", "typing", "file", "run", "suggestion"]
    duration_ms: float = Field(ge=0, le=600_000)
    epoch_ms: float = Field(ge=0)


class Control(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Literal["local_off", "local_on", "latency_250", "latency_1000"]
    delay_ms: Literal[0, 250, 1000] = 0


def attach(app, root):
    log = Path(root) / "qa-performance.jsonl"
    lock = threading.Lock()
    settings = {"scenario": "local_off", "delay_ms": 0}
    count = 0

    @app.middleware("http")
    async def latency(request, next_):
        if settings["delay_ms"] and request.url.path.startswith(
            ("/api/desktop/projects/", "/api/desktop/local-projects")
        ):
            await asyncio.sleep(settings["delay_ms"] / 1000)
        return await next_(request)

    @app.post("/api/desktop/qa-performance/control")
    def control(body: Control):
        settings.update(body.model_dump())
        return dict(settings)

    @app.post("/api/desktop/qa-performance/event")
    def record(body: Event):
        nonlocal count
        with lock:
            if count >= 5000:
                raise HTTPException(413, "QA event budget reached.")
            # Only labels/times; no code, keystrokes, project names or credentials.
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**body.model_dump(), **settings}, allow_nan=False) + "\n")
            count += 1
        return {"recorded": True}
