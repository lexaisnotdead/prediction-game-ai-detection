from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scoring import MAX_STREAM_DELAY_SECONDS, Prediction, clamp_stream_delay, score


class ScoringTests(unittest.TestCase):
    def test_clamp_stream_delay_limits_untrusted_values(self) -> None:
        self.assertEqual(clamp_stream_delay(-5), 0.0)
        self.assertEqual(clamp_stream_delay(999), MAX_STREAM_DELAY_SECONDS)

    def test_rejects_prediction_that_arrives_after_detected_event(self) -> None:
        result = score(
            Prediction(
                client_id="client-a",
                match_id="football",
                event_type="goal",
                predicted_offset=10.0,
                stream_delay=2.0,
                server_recv_time=100.5,
            ),
            event_ts=12.2,
            event_type="goal",
            event_received_time=100.4,
        )

        self.assertTrue(result.rejected)
        self.assertEqual(result.reject_reason, "prediction arrived after detected event")

    def test_rejects_client_timing_that_exceeds_server_observed_lead(self) -> None:
        result = score(
            Prediction(
                client_id="client-a",
                match_id="football",
                event_type="goal",
                predicted_offset=50.0,
                stream_delay=2.0,
                server_recv_time=100.0,
            ),
            event_ts=54.6,
            event_type="goal",
            event_received_time=101.0,
        )

        self.assertTrue(result.rejected)
        self.assertEqual(result.reject_reason, "client timing exceeds server-observed lead")

    def test_scores_when_client_timing_matches_server_lead(self) -> None:
        result = score(
            Prediction(
                client_id="client-a",
                match_id="basketball",
                event_type="goal",
                predicted_offset=120.0,
                stream_delay=2.2,
                server_recv_time=200.0,
            ),
            event_ts=123.3,
            event_type="goal",
            event_received_time=201.0,
        )

        self.assertFalse(result.rejected)
        self.assertEqual(result.pts, 500)
        self.assertAlmostEqual(result.delta_server, 1.0)


if __name__ == "__main__":
    unittest.main()
