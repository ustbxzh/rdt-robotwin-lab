"""Env↔policy transport over WebSocket (default) and legacy TCP.

The upstream runtime imports ``client_server`` as a top-level package.  Keep a
compatibility alias so the curated package can live under ``policy`` without
rewriting the protocol implementation.
"""

from __future__ import annotations

import sys
from typing import Any

sys.modules.setdefault("client_server", sys.modules[__name__])

__all__ = [
    "ModelClient",
    "ModelServer",
    "ModelServerConfig",
    "PolicyServer",
    "PolicyServerConfig",
    "WsModelClient",
]


def __getattr__(name: str) -> Any:
    if name in ("ModelClient", "WsModelClient"):
        from client_server.ws.model_client import WsModelClient

        return WsModelClient
    if name in ("ModelServer", "PolicyServer"):
        from client_server.ws.model_server import PolicyServer

        if name == "ModelServer":
            return PolicyServer
        return PolicyServer
    if name in ("ModelServerConfig", "PolicyServerConfig"):
        from client_server.ws.model_server import PolicyServerConfig

        if name == "ModelServerConfig":
            return PolicyServerConfig
        return PolicyServerConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
