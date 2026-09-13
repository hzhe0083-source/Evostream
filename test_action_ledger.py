"""Unit tests for ExecutedActionLedger and evaluate.py integration."""

from __future__ import annotations

import threading
import unittest
from unittest.mock import patch

import numpy as np

from streaming import ExecutedActionLedger


class TestActionLedger(unittest.TestCase):
    def test_record_basic_and_command_id(self) -> None:
        ledger = ExecutedActionLedger(capacity=10)
        action = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        rec = ledger.record(
            action,
            timestamp=10.0,
            duration=0.1,
            step_index=5,
            plan_id=None,
            event_epoch=0,
            fallback=False,
        )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger.total_recorded, 1)
        self.assertEqual(rec.step_index, 5)
        self.assertEqual(rec.command_id, 0)
        self.assertEqual(rec.timestamp, 10.0)
        self.assertEqual(rec.duration, 0.1)
        self.assertIsNone(rec.plan_id)
        self.assertEqual(rec.event_epoch, 0)
        self.assertFalse(rec.fallback)
        np.testing.assert_allclose(rec.action, action)

        # Second record increments command_id independently of step_index
        rec2 = ledger.record(
            action,
            timestamp=10.2,
            duration=0.1,
            step_index=6,
        )
        self.assertEqual(rec2.command_id, 1)
        self.assertEqual(rec2.step_index, 6)
        self.assertEqual(ledger.total_recorded, 2)

    def test_immutability_and_defensive_copying(self) -> None:
        ledger = ExecutedActionLedger()
        action = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        rec = ledger.record(action, timestamp=1.0, duration=0.1)

        # Mutating original array does not change recorded action
        action[0] = 999.0
        self.assertEqual(rec.action[0], 1.0)
        records = ledger.get_records()
        self.assertEqual(records[0].action[0], 1.0)

        # Mutating array obtained through get_records should fail or not affect ledger
        with self.assertRaises(ValueError):
            records[0].action[0] = 888.0

        # Also verify copy method produces an independent copy
        copied = rec.copy()
        self.assertEqual(copied.action[0], 1.0)

    def test_environment_mutation_simulation(self) -> None:
        """Simulate env.step mutating the passed action in place."""
        ledger = ExecutedActionLedger()
        action = np.array([0.5, -0.5, 1.0], dtype=np.float32)

        # Emulating the snapshot pattern in evaluate.py
        action_snapshot = action.copy()

        # Simulated env.step modifies action in-place
        def fake_env_step(act: np.ndarray) -> None:
            act[:] = 0.0

        fake_env_step(action)

        rec = ledger.record(
            action_snapshot,
            timestamp=1.0,
            duration=0.01,
            step_index=0,
            plan_id=None,
        )
        self.assertEqual(rec.action[0], 0.5)
        self.assertEqual(action[0], 0.0)

    def test_blocking_boundary_records_final_gripper_before_env_mutation(self) -> None:
        from evaluate import _blocking_rollout

        submitted = []

        class Env:
            def step(self, action):
                submitted.append(action.copy())
                action[:] = 0
                return {}, 0, False, {}

            def check_success(self):
                return True

        ledger = ExecutedActionLedger()
        with patch("evaluate._image", return_value=None), patch(
            "evaluate._robot_state", return_value=np.zeros(2)
        ):
            success, steps, _ = _blocking_rollout(
                Env(), {}, lambda obs: None,
                lambda obs: {"actions": np.array([[0.3, -0.4, 0.0, 0.0, 0.0, 0.0, 0.1]], dtype=np.float32), "visual_age": 0.0},
                {}, 1, 1, None, 0.1, action_ledger=ledger,
            )
        self.assertTrue(success)
        self.assertEqual(steps, 1)
        record = ledger.get_records()[0]
        np.testing.assert_array_equal(record.action, submitted[0])
        self.assertEqual(record.action[-1], -1.0)
        self.assertIsNone(record.plan_id)

    def test_invalid_values_validation(self) -> None:
        ledger = ExecutedActionLedger()

        # Non-finite action
        with self.assertRaises(ValueError):
            ledger.record(np.array([np.nan, 0.0]))
        with self.assertRaises(ValueError):
            ledger.record(np.array([np.inf, 0.0]))

        # Non-finite timestamp
        with self.assertRaises(ValueError):
            ledger.record(np.array([1.0, 2.0]), timestamp=float("nan"))
        with self.assertRaises(ValueError):
            ledger.record(np.array([1.0, 2.0]), timestamp=float("inf"))

        # Negative or non-finite duration
        with self.assertRaises(ValueError):
            ledger.record(np.array([1.0, 2.0]), timestamp=1.0, duration=-0.01)
        with self.assertRaises(ValueError):
            ledger.record(np.array([1.0, 2.0]), timestamp=1.0, duration=float("nan"))

    def test_monotonicity_and_boundary_overlap(self) -> None:
        ledger = ExecutedActionLedger()
        ledger.record(np.array([1.0]), timestamp=10.0, duration=0.1)

        # Monotonicity violation: t < last.timestamp
        with self.assertRaises(ValueError) as ctx:
            ledger.record(np.array([1.0]), timestamp=9.9, duration=0.1)
        self.assertIn("monotonicity violation", str(ctx.exception))

        # Overlapping boundary: t < last.timestamp + last.duration
        with self.assertRaises(ValueError) as ctx:
            ledger.record(np.array([1.0]), timestamp=10.05, duration=0.1)
        self.assertIn("overlapping boundary", str(ctx.exception))

        # Exactly at boundary t == last.timestamp + last.duration is valid
        rec2 = ledger.record(np.array([2.0]), timestamp=10.1, duration=0.1)
        self.assertEqual(rec2.command_id, 1)

    def test_bounded_capacity_and_truncation(self) -> None:
        ledger = ExecutedActionLedger(capacity=3)
        for i in range(5):
            ledger.record(np.array([float(i)]), timestamp=float(i), duration=1.0)
        self.assertEqual(len(ledger), 3)
        self.assertEqual(ledger.total_recorded, 5)

        records = ledger.get_records()
        self.assertEqual([r.command_id for r in records], [2, 3, 4])

        # Querying an interval before retained records detects truncation
        res = ledger.query_interval(0.5, 1.5)
        self.assertTrue(res["truncated"])
        self.assertFalse(res["covered"])

        # Querying within retained window has no truncation
        res2 = ledger.query_interval(2.5, 4.5)
        self.assertFalse(res2["truncated"])
        self.assertTrue(res2["covered"])

    def test_query_interval_internal_gap_detection(self) -> None:
        ledger = ExecutedActionLedger()
        # Segment 1: [10.0, 10.2)
        ledger.record(np.array([1.0]), timestamp=10.0, duration=0.2)
        # Gap: [10.2, 10.5) has no actions
        # Segment 2: [10.5, 10.7)
        ledger.record(np.array([2.0]), timestamp=10.5, duration=0.2)

        # Query [10.0, 10.7] covers the outer endpoints, but contains an internal gap
        res = ledger.query_interval(10.0, 10.7)
        self.assertEqual(len(res["records"]), 2)
        self.assertFalse(res["covered"])  # Internal gap must cause covered=False
        self.assertFalse(res["truncated"])

        # Query strictly within Segment 1: [10.05, 10.15]
        res1 = ledger.query_interval(10.05, 10.15)
        self.assertEqual(len(res1["records"]), 1)
        self.assertTrue(res1["covered"])

        # Query strictly within Segment 2: [10.5, 10.7]
        res2 = ledger.query_interval(10.5, 10.7)
        self.assertEqual(len(res2["records"]), 1)
        self.assertTrue(res2["covered"])

    def test_query_interval_point_query(self) -> None:
        ledger = ExecutedActionLedger()
        ledger.record(np.array([1.0]), timestamp=1.0, duration=0.5)

        # Point inside interval
        res = ledger.query_interval(1.2, 1.2)
        self.assertTrue(res["covered"])
        self.assertEqual(len(res["records"]), 1)

        # Point outside interval
        res2 = ledger.query_interval(2.0, 2.0)
        self.assertFalse(res2["covered"])
        self.assertEqual(len(res2["records"]), 0)

    def test_stats_and_export(self) -> None:
        ledger = ExecutedActionLedger()
        empty_stats = ledger.stats()
        self.assertEqual(empty_stats["total_recorded"], 0)
        self.assertEqual(empty_stats["retained_count"], 0)
        self.assertEqual(empty_stats["fallback_fraction"], 0.0)

        ledger.record(np.array([1.0]), timestamp=1.0, duration=0.1, fallback=False)
        ledger.record(np.array([2.0]), timestamp=1.2, duration=0.1, fallback=True)

        stats = ledger.stats()
        self.assertEqual(stats["total_recorded"], 2)
        self.assertEqual(stats["retained_count"], 2)
        self.assertEqual(stats["fallback_count"], 1)
        self.assertAlmostEqual(stats["fallback_fraction"], 0.5)
        self.assertAlmostEqual(stats["mean_duration"], 0.1)

        exported = ledger.export()
        self.assertEqual(len(exported), 2)
        self.assertEqual(exported[0]["command_id"], 0)
        self.assertEqual(exported[0]["action"], [1.0])
        self.assertFalse(exported[0]["fallback"])
        self.assertEqual(exported[1]["command_id"], 1)
        self.assertTrue(exported[1]["fallback"])

    def test_thread_safety(self) -> None:
        ledger = ExecutedActionLedger(capacity=1000)
        num_threads = 4
        records_per_thread = 50

        # To test thread-safety with monotonicity, serialize timestamps or use a shared clock
        lock = threading.Lock()
        cur_t = [1.0]

        def worker(thread_idx: int) -> None:
            for _ in range(records_per_thread):
                with lock:
                    t = cur_t[0]
                    cur_t[0] += 0.01
                    ledger.record(
                        np.array([float(thread_idx)]),
                        timestamp=t,
                        duration=0.01,
                    )

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(num_threads)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        self.assertEqual(len(ledger), num_threads * records_per_thread)
        self.assertEqual(ledger.total_recorded, num_threads * records_per_thread)
        records = ledger.get_records()
        for i in range(len(records) - 1):
            self.assertLessEqual(records[i].timestamp + records[i].duration, records[i + 1].timestamp + 1e-9)


if __name__ == "__main__":
    unittest.main()
