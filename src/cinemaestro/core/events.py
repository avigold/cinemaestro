"""Pipeline event system for progress reporting and logging."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine


class EventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGE_ERROR = "stage_error"
    SHOT_GENERATION_START = "shot_generation_start"
    SHOT_GENERATION_COMPLETE = "shot_generation_complete"
    SHOT_GENERATION_ERROR = "shot_generation_error"
    AUDIO_GENERATION_START = "audio_generation_start"
    AUDIO_GENERATION_COMPLETE = "audio_generation_complete"
    CONSISTENCY_CHECK_START = "consistency_check_start"
    CONSISTENCY_CHECK_RESULT = "consistency_check_result"
    COST_UPDATE = "cost_update"
    PROGRESS = "progress"


@dataclass
class PipelineEvent:
    event_type: EventType
    stage: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    progress: float = 0.0  # 0.0 to 1.0


EventHandler = Callable[[PipelineEvent], Coroutine[Any, Any, None]]


class EventBus:
    """Simple async event bus for pipeline progress and logging."""

    def __init__(self) -> None:
        self._handlers: dict[EventType | None, list[EventHandler]] = {}

    def on(
        self, event_type: EventType | None = None
    ) -> Callable[[EventHandler], EventHandler]:
        """Decorator to register an event handler. None = all events."""

        def decorator(handler: EventHandler) -> EventHandler:
            self._handlers.setdefault(event_type, []).append(handler)
            return handler

        return decorator

    def subscribe(
        self, handler: EventHandler, event_type: EventType | None = None
    ) -> None:
        self._handlers.setdefault(event_type, []).append(handler)

    async def emit(self, event: PipelineEvent) -> None:
        handlers = self._handlers.get(event.event_type, []) + self._handlers.get(
            None, []
        )
        await asyncio.gather(*(h(event) for h in handlers), return_exceptions=True)
