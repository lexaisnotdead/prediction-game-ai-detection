from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from game_session import ConnectionRegistry, PendingPredictions


class FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.messages: list[dict] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, message: dict) -> None:
        self.messages.append(message)


class ConnectionRegistryTests(unittest.IsolatedAsyncioTestCase):
    async def test_send_to_client_only_hits_target_client(self) -> None:
        registry = ConnectionRegistry()
        ws_a = FakeWebSocket()
        ws_b = FakeWebSocket()

        await registry.connect("football", "client-a", ws_a)
        await registry.connect("football", "client-b", ws_b)
        await registry.send_to_client("football", "client-a", {"type": "score_result", "pts": 500})

        self.assertEqual(ws_a.messages, [{"type": "score_result", "pts": 500}])
        self.assertEqual(ws_b.messages, [])

    async def test_disconnect_invalidates_prediction_token(self) -> None:
        registry = ConnectionRegistry()
        ws = FakeWebSocket()

        token = await registry.connect("basketball", "client-a", ws)
        self.assertIsNotNone(registry.get_session(token))

        registry.disconnect("basketball", ws)
        self.assertIsNone(registry.get_session(token))


class PendingPredictionsTests(unittest.TestCase):
    def test_add_replaces_existing_prediction_for_same_client(self) -> None:
        pending = PendingPredictions()

        first_insert = pending.add("football", "client-a", {"id": 1})
        second_insert = pending.add("football", "client-a", {"id": 2})
        pending.add("football", "client-b", {"id": 3})

        self.assertTrue(first_insert)
        self.assertFalse(second_insert)
        self.assertEqual(
            pending.pop_match("football"),
            [{"id": 2}, {"id": 3}],
        )


if __name__ == "__main__":
    unittest.main()
