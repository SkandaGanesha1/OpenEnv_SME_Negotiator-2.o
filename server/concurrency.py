"""Limit concurrent /reset and /step requests (parallel eval safety)."""

from __future__ import annotations

import asyncio
import os
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


def max_concurrent_envs_from_env() -> int:
    raw = os.getenv("MAX_CONCURRENT_ENVS", "4")
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


class OpenEnvConcurrencyLimiter(BaseHTTPMiddleware):
    """Return 503 when too many simulation requests run at once."""

    _LIMIT_PATHS = frozenset({"/reset", "/step"})

    def __init__(self, app: Callable, *, max_concurrent: int) -> None:
        super().__init__(app)
        self._max = max_concurrent
        self._lock = asyncio.Lock()
        self._active = 0

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path not in self._LIMIT_PATHS:
            return await call_next(request)

        async with self._lock:
            if self._active >= self._max:
                return JSONResponse(
                    status_code=503,
                    content={
                        "detail": (
                            "Server at maximum concurrent OpenEnv capacity "
                            f"({self._max} in-flight reset/step operations). Retry shortly."
                        ),
                        "error": "capacity_exceeded",
                        "max_concurrent_envs": self._max,
                    },
                )
            self._active += 1

        try:
            return await call_next(request)
        finally:
            async with self._lock:
                self._active -= 1
