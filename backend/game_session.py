from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from uuid import uuid4


@dataclass(frozen=True)
class PredictionSession:
    match_id: str
    client_id: str
    token: str


class ConnectionRegistry:
    def __init__(self) -> None:
        self._connections: dict[str, dict[object, PredictionSession]] = {}
        self._tokens: dict[str, PredictionSession] = {}

    async def connect(self, match_id: str, client_id: str, ws: object) -> str:
        await ws.accept()
        token = uuid4().hex
        session = PredictionSession(match_id=match_id, client_id=client_id, token=token)
        self._connections.setdefault(match_id, {})[ws] = session
        self._tokens[token] = session
        return token

    def disconnect(self, match_id: str, ws: object) -> None:
        match_connections = self._connections.get(match_id)
        if not match_connections:
            return

        session = match_connections.pop(ws, None)
        if not match_connections:
            self._connections.pop(match_id, None)
        if session is not None:
            self._tokens.pop(session.token, None)

    def count(self, match_id: str) -> int:
        return len(self._connections.get(match_id, {}))

    def get_session(self, token: str) -> Optional[PredictionSession]:
        return self._tokens.get(token)

    async def broadcast(self, match_id: str, message: dict) -> None:
        dead: list[object] = []
        for ws in self._connections.get(match_id, {}):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(match_id, ws)

    async def send_to_client(self, match_id: str, client_id: str, message: dict) -> None:
        dead: list[object] = []
        for ws, session in self._connections.get(match_id, {}).items():
            if session.client_id != client_id:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)

        for ws in dead:
            self.disconnect(match_id, ws)


class PendingPredictions:
    def __init__(self) -> None:
        self._pending: dict[str, dict[str, object]] = {}

    def add(self, match_id: str, client_id: str, prediction: object) -> bool:
        match_pending = self._pending.setdefault(match_id, {})
        replaced = client_id in match_pending
        match_pending[client_id] = prediction
        return not replaced

    def pop_match(self, match_id: str) -> list[object]:
        return list(self._pending.pop(match_id, {}).values())
