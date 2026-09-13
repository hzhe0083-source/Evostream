"""Tests and verification for P2 Sentinel modules."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from sentinel import (
    DtBinThreshold,
    FixedFeatureNormalizer,
    LatentDriftMonitor,
    SentinelCalibrationProfile,
    ShallowDynamicsPredictor,
    huber_residual,
)
from sentinel_data import (
    ContextSpec,
    FeatureSpec,
    SentinelTransition,
    SentinelTransitionDataset,
    TimingContract,
    TransitionContractMetadata,
    extract_transitions_from_recordings,
    verify_disjoint_episodes,
)
from train_sentinel import (
    calibrate_sentinel,
    compute_predictor_state_hash,
    load_sentinel_predictor,
    replay_shadow_transitions,
    train_sentinel,
)


def _make_metadata(
    num_regions: int = 4,
    feature_dim: int = 16,
    context_dim: int = 8,
    frame_interval: float = 0.1,
    action_interval: float = 0.05,
    policy_hash: str = "policy_hash_test_123",
    shallow_weights_fingerprint: str = "shallow_fp_abc_001",
) -> TransitionContractMetadata:
    return TransitionContractMetadata(
        policy_hash=policy_hash,
        vision_source_fingerprint="vision_source_fp_456",
        preprocessing_fingerprint="crop_resize_norm_fp_789",
        shallow_weights_fingerprint=shallow_weights_fingerprint,
        split_depth="layer_24",
        feature_spec=FeatureSpec(
            num_regions=num_regions,
            feature_dim=feature_dim,
            grid_h=2,
            grid_w=2,
        ),
        timing_contract=TimingContract(
            frame_interval=frame_interval,
            action_interval=action_interval,
        ),
        context_spec=ContextSpec(
            context_dim=context_dim,
            context_method="held_accepted_prefix",
        ),
    )


def _make_recording(
    episode_id: str,
    num_frames: int = 10,
    num_regions: int = 4,
    feature_dim: int = 16,
    state_dim: int = 6,
    action_dim: int = 4,
    context_dim: int = 8,
    frame_interval: float = 0.1,
    action_interval: float = 0.05,
    context_ts: float = 0.0,
    unequal_action_durations: bool = False,
) -> dict[str, Any]:
    timestamps = torch.arange(num_frames, dtype=torch.float64) * frame_interval
    frozen_features = torch.randn(num_frames, num_regions, feature_dim)
    states = torch.randn(num_frames, state_dim)
    velocities = torch.randn(num_frames, state_dim)

    total_time = (num_frames - 1) * frame_interval + 0.5
    start_ts_list = []
    end_ts_list = []
    curr = 0.0
    idx = 0
    while curr < total_time:
        dur = 0.03 if (unequal_action_durations and idx % 2 == 0) else action_interval
        start_ts_list.append(curr)
        end_ts_list.append(curr + dur)
        curr += dur
        idx += 1

    actions = torch.randn(len(start_ts_list), action_dim)
    action_timestamps = torch.tensor(start_ts_list, dtype=torch.float64)
    action_end_timestamps = torch.tensor(end_ts_list, dtype=torch.float64)

    context = torch.randn(context_dim)
    context_timestamps = torch.tensor(context_ts, dtype=torch.float64)

    return {
        "episode_id": episode_id,
        "timestamps": timestamps,
        "frozen_features": frozen_features,
        "states": states,
        "velocities": velocities,
        "actions": actions,
        "action_timestamps": action_timestamps,
        "action_end_timestamps": action_end_timestamps,
        "context": context,
        "context_timestamps": context_timestamps,
    }


class SentinelContractTests(unittest.TestCase):
    def test_fixed_feature_normalizer_has_no_parameters(self) -> None:
        norm = FixedFeatureNormalizer(feature_dim=16)
        self.assertEqual(len(list(norm.parameters())), 0)
        self.assertEqual(len(list(norm.buffers())), 2)

        x = torch.randn(2, 4, 16)
        x_norm = norm(x)
        torch.testing.assert_close(x, norm.denormalize(x_norm))

    def test_spatial_region_coordinate_buffer_registered(self) -> None:
        pred = ShallowDynamicsPredictor(
            num_regions=4,
            feature_dim=16,
            state_dim=6,
            action_dim=4,
            context_dim=8,
            grid_h=2,
            grid_w=2,
        )
        self.assertTrue(hasattr(pred, "region_coords"))
        self.assertEqual(pred.region_coords.shape, (4, 2))
        self.assertTrue(torch.all(pred.region_coords >= -1.0))
        self.assertTrue(torch.all(pred.region_coords <= 1.0))

    def test_predictor_never_accepts_target_z_or_state(self) -> None:
        import inspect

        sig = inspect.signature(ShallowDynamicsPredictor.forward)
        params = list(sig.parameters.keys())
        for forbidden in ("target_z", "target_state", "end_z", "end_state"):
            self.assertNotIn(forbidden, params)

    def test_duration_weighted_action_summary_unequal_durations(self) -> None:
        meta = _make_metadata()
        rec = _make_recording("ep_unequal", unequal_action_durations=True)
        transitions = extract_transitions_from_recordings([rec], meta, spans=[1, 2])
        self.assertGreater(len(transitions), 0)
        t = transitions[0]
        self.assertTrue(torch.all(torch.isfinite(t.action_mean)))
        self.assertTrue(torch.all(torch.isfinite(t.action_last)))
        self.assertGreater(t.end_time, t.start_time)
        overlaps = (torch.minimum(rec["action_end_timestamps"], torch.tensor(t.end_time))
                    - torch.maximum(rec["action_timestamps"], torch.tensor(t.start_time))).clamp_min(0)
        expected = (rec["actions"].double() * overlaps[:, None]).sum(0) / (t.end_time - t.start_time)
        torch.testing.assert_close(t.action_mean.double(), expected, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(t.action_last, rec["actions"][torch.nonzero(overlaps > 0)[-1, 0]])

    def test_cross_boundary_action_and_future_features(self) -> None:
        meta = _make_metadata()
        rec = _make_recording("cross", num_frames=2)
        rec["timestamps"] = torch.tensor([1000000.0, 1000000.1], dtype=torch.float64)
        rec["action_timestamps"] = torch.tensor([999999.9, 1000000.025], dtype=torch.float64)
        rec["action_end_timestamps"] = torch.tensor([1000000.025, 1000000.2], dtype=torch.float64)
        rec["actions"] = torch.tensor([[0., 0., 0., 0.], [4., 4., 4., 4.]])
        before = extract_transitions_from_recordings([rec], meta, spans=[1])[0]
        torch.testing.assert_close(before.action_mean, torch.full((4,), 3.0))
        rec["frozen_features"][1] += 100
        after = extract_transitions_from_recordings([rec], meta, spans=[1])[0]
        for name in ("start_z", "start_state", "start_velocity", "action_mean", "action_last", "task_context"):
            torch.testing.assert_close(getattr(before, name), getattr(after, name))

    def test_anchor_rejects_backwards_time_and_nonfinite_thresholds(self) -> None:
        from sentinel import AnchorPersistenceTracker
        tracker = AnchorPersistenceTracker("short", 0.1)
        tracker.update(True, 1.0)
        with self.assertRaises(ValueError):
            tracker.update(True, 0.9)
        with self.assertRaises(ValueError):
            DtBinThreshold(0.1, 0.2, float("nan"), 1.0)

    def test_extractor_rejects_gaps_and_missing_coverage(self) -> None:
        meta = _make_metadata()
        rec = _make_recording("ep_gap")
        rec["action_timestamps"][2] += 0.2
        rec["action_end_timestamps"][2] += 0.2
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec], meta, spans=[2])

    def test_temporal_context_no_future_leakage(self) -> None:
        meta = _make_metadata()
        rec = _make_recording("ep_leak", context_ts=10.0)
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec], meta, spans=[1])

    def test_disjoint_episodes_validation(self) -> None:
        meta = _make_metadata()
        rec1 = _make_recording("ep1")
        rec2 = _make_recording("ep2")
        rec3 = _make_recording("ep1")

        t1 = extract_transitions_from_recordings([rec1], meta, spans=[1])
        t2 = extract_transitions_from_recordings([rec2], meta, spans=[1])
        t3 = extract_transitions_from_recordings([rec3], meta, spans=[1])

        verify_disjoint_episodes(t1, t2)
        with self.assertRaises(ValueError):
            verify_disjoint_episodes(t1, t3)

    def test_context_time_lookup_sparse_updates(self) -> None:
        meta = _make_metadata(context_dim=2)
        rec = _make_recording("ep_sparse_ctx", num_frames=5, frame_interval=0.1, context_dim=2)
        rec["context"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        rec["context_timestamps"] = torch.tensor([0.0, 0.3], dtype=torch.float64)

        transitions = extract_transitions_from_recordings([rec], meta, spans=[1])
        t_by_start = {round(t.start_time, 4): t for t in transitions}

        torch.testing.assert_close(t_by_start[0.0].task_context, torch.tensor([1.0, 2.0]))
        self.assertAlmostEqual(t_by_start[0.0].context_age.item(), 0.0)

        torch.testing.assert_close(t_by_start[0.1].task_context, torch.tensor([1.0, 2.0]))
        self.assertAlmostEqual(t_by_start[0.1].context_age.item(), 0.1)

        torch.testing.assert_close(t_by_start[0.2].task_context, torch.tensor([1.0, 2.0]))
        self.assertAlmostEqual(t_by_start[0.2].context_age.item(), 0.2)

        torch.testing.assert_close(t_by_start[0.3].task_context, torch.tensor([3.0, 4.0]))
        self.assertAlmostEqual(t_by_start[0.3].context_age.item(), 0.0)

    def test_context_future_mutation_isolation(self) -> None:
        meta = _make_metadata(context_dim=2)
        rec = _make_recording("ep_mut", num_frames=3, context_dim=2)
        ctx_tensor = torch.tensor([[5.0, 6.0]], dtype=torch.float32)
        rec["context"] = ctx_tensor
        rec["context_timestamps"] = torch.tensor([0.0], dtype=torch.float64)

        transitions = extract_transitions_from_recordings([rec], meta, spans=[1])
        ctx_tensor[0, 0] = 999.0
        self.assertEqual(transitions[0].task_context[0].item(), 5.0)

    def test_context_validation_errors_and_duplicates(self) -> None:
        meta = _make_metadata(context_dim=2)
        rec = _make_recording("ep_val", num_frames=3, context_dim=2)

        rec_bad_count = dict(rec)
        rec_bad_count["context"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rec_bad_count["context_timestamps"] = torch.tensor([0.0])
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_bad_count], meta, spans=[1])

        rec_bad_order = dict(rec)
        rec_bad_order["context"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rec_bad_order["context_timestamps"] = torch.tensor([0.2, 0.1])
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_bad_order], meta, spans=[1])

        rec_contra = dict(rec)
        rec_contra["context"] = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rec_contra["context_timestamps"] = torch.tensor([0.0, 0.0])
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_contra], meta, spans=[1])

        rec_compat = dict(rec)
        rec_compat["context"] = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
        rec_compat["context_timestamps"] = torch.tensor([0.0, 0.0])
        t_compat = extract_transitions_from_recordings([rec_compat], meta, spans=[1])
        self.assertGreater(len(t_compat), 0)

        rec_no_prior = dict(rec)
        rec_no_prior["context"] = torch.tensor([[1.0, 2.0]])
        rec_no_prior["context_timestamps"] = torch.tensor([5.0])
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_no_prior], meta, spans=[1])

    def test_action_interval_strict_overlap_tiny_boundary_command(self) -> None:
        meta = _make_metadata()
        rec = {
            "episode_id": "ep_boundary",
            "timestamps": torch.tensor([0.0, 0.1000], dtype=torch.float64),
            "frozen_features": torch.randn(2, 4, 16),
            "states": torch.randn(2, 6),
            "velocities": torch.randn(2, 6),
            "actions": torch.tensor([[1.0, 0.0], [0.0, 10.0]], dtype=torch.float32),
            "action_timestamps": torch.tensor([0.0, 0.0999], dtype=torch.float64),
            "action_end_timestamps": torch.tensor([0.0999, 0.1001], dtype=torch.float64),
            "context": torch.randn(8),
            "context_timestamps": torch.tensor(0.0, dtype=torch.float64),
        }
        transitions = extract_transitions_from_recordings([rec], meta, spans=[1], time_tolerance=1e-4)
        self.assertEqual(len(transitions), 1)
        trans = transitions[0]
        torch.testing.assert_close(trans.action_last, torch.tensor([0.0, 10.0]))
        expected_mean = (torch.tensor([1.0, 0.0]) * 0.0999 + torch.tensor([0.0, 10.0]) * 0.0001) / 0.1000
        torch.testing.assert_close(trans.action_mean, expected_mean, atol=1e-5, rtol=1e-5)

    def test_extract_shape_finite_and_tolerance_checks(self) -> None:
        meta = _make_metadata(num_regions=4, feature_dim=16, context_dim=8)
        rec = _make_recording("ep_checks", num_regions=4, feature_dim=16, context_dim=8)

        # Invalid time_tolerance
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec], meta, spans=[1], time_tolerance=-0.01)
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec], meta, spans=[1], time_tolerance=float("nan"))

        # Non-finite actions
        rec_bad_act = dict(rec)
        rec_bad_act["actions"] = rec["actions"].clone()
        rec_bad_act["actions"][0, 0] = float("nan")
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_bad_act], meta, spans=[1])

        # Action count mismatch with timestamps
        rec_bad_shape = dict(rec)
        rec_bad_shape["action_timestamps"] = rec["action_timestamps"][:-1]
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_bad_shape], meta, spans=[1])

        # Non-strictly increasing action timestamps
        rec_descending_acts = dict(rec)
        rec_descending_acts["action_timestamps"] = rec["action_timestamps"].clone()
        rec_descending_acts["action_timestamps"][1] = rec_descending_acts["action_timestamps"][0]
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_descending_acts], meta, spans=[1])

        # Feature spec trailing dimensions mismatch
        rec_bad_feat = dict(rec)
        rec_bad_feat["frozen_features"] = torch.randn(len(rec["timestamps"]), 4, 32)
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_bad_feat], meta, spans=[1])

        # State/velocity shape mismatch
        rec_bad_vel = dict(rec)
        rec_bad_vel["velocities"] = torch.randn(len(rec["timestamps"]), 3)
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_bad_vel], meta, spans=[1])

        # Context dimension mismatch
        rec_bad_ctx_dim = dict(rec)
        rec_bad_ctx_dim["context"] = torch.randn(4)  # expected 8
        with self.assertRaises(ValueError):
            extract_transitions_from_recordings([rec_bad_ctx_dim], meta, spans=[1])

    def test_malformed_string_episode_lists_rejected(self) -> None:
        meta = _make_metadata()
        b0 = DtBinThreshold(dt_min=0.08, dt_max=0.12, region_quantile_threshold=1.0, global_threshold=1.0)
        valid_dict = {
            "bins": [{"dt_min": 0.08, "dt_max": 0.12, "region_quantile_threshold": 1.0, "global_threshold": 1.0}],
            "persistence_seconds": 0.5,
            "quantile": 0.9,
            "quantile_weight": 0.5,
            "metadata": meta.to_dict(),
            "predictor_state_hash": "pred_hash_123",
            "calibration_episodes": "scalar_string_should_be_list",  # Scalar string instead of list!
        }
        with self.assertRaises(ValueError):
            SentinelCalibrationProfile.from_dict(valid_dict)

        with self.assertRaises(ValueError):
            SentinelCalibrationProfile(
                bins=(b0,),
                persistence_seconds=0.5,
                quantile=0.9,
                quantile_weight=0.5,
                metadata=meta,
                predictor_state_hash="pred_hash_123",
                calibration_episodes="not_a_tuple",  # type: ignore[arg-type]
            )

    def test_persistence_zero_does_not_alert_on_normal(self) -> None:
        meta = _make_metadata()
        pred = ShallowDynamicsPredictor(
            num_regions=4,
            feature_dim=16,
            state_dim=6,
            action_dim=4,
            context_dim=8,
            grid_h=2,
            grid_w=2,
            hidden_dim=32,
        )
        b0 = DtBinThreshold(dt_min=0.08, dt_max=0.12, region_quantile_threshold=1.0, global_threshold=1.0)
        profile = SentinelCalibrationProfile(
            bins=(b0,),
            persistence_seconds=0.0,
            quantile=0.90,
            quantile_weight=0.5,
            metadata=meta,
            predictor_state_hash="dummy_pred_hash",
            calibration_episodes=("ep_cal_0",),
        )
        monitor = LatentDriftMonitor(predictor=pred, profile=profile)

        start_z = torch.randn(4, 16)
        clean_target = start_z.clone()
        res = monitor.evaluate_step(
            start_z=start_z,
            target_z=clean_target,
            start_state=torch.zeros(6),
            start_velocity=torch.zeros(6),
            action_mean=torch.zeros(4),
            action_last=torch.zeros(4),
            dt=0.1,
            task_context=torch.zeros(8),
            context_age=0.0,
            timestamp=0.0,
        )
        self.assertFalse(res["is_step_drift"])
        self.assertFalse(res["persistent_alert"])

    def test_short_and_long_anchor_persistence_tracking(self) -> None:
        meta = _make_metadata()
        pred = ShallowDynamicsPredictor(
            num_regions=4,
            feature_dim=16,
            state_dim=6,
            action_dim=4,
            context_dim=8,
            grid_h=2,
            grid_w=2,
            hidden_dim=32,
        )
        # Adjacent bins sorted [0.08, 0.2) and [0.2, 0.5)
        b_short = DtBinThreshold(dt_min=0.08, dt_max=0.2, region_quantile_threshold=0.01, global_threshold=0.01)
        b_long = DtBinThreshold(dt_min=0.2, dt_max=0.5, region_quantile_threshold=0.01, global_threshold=0.01)
        profile = SentinelCalibrationProfile(
            bins=(b_short, b_long),
            persistence_seconds=0.2,
            quantile=0.90,
            quantile_weight=0.5,
            metadata=meta,
            predictor_state_hash="dummy_hash",
            calibration_episodes=("ep_cal_0",),
        )
        monitor = LatentDriftMonitor(
            predictor=pred,
            profile=profile,
            anchor_persistence_seconds={"short_span": 0.1, "long_span": 0.5},
        )

        start_z = torch.randn(4, 16)
        drift_target = start_z + 10.0

        res_s1 = monitor.evaluate_step(
            start_z=start_z,
            target_z=drift_target,
            start_state=torch.zeros(6),
            start_velocity=torch.zeros(6),
            action_mean=torch.zeros(4),
            action_last=torch.zeros(4),
            dt=0.1,
            task_context=torch.zeros(8),
            context_age=0.0,
            timestamp=0.0,
            anchor="short_span",
        )
        self.assertTrue(res_s1["is_step_drift"])
        self.assertFalse(res_s1["persistent_alert"])

        res_s2 = monitor.evaluate_step(
            start_z=start_z,
            target_z=drift_target,
            start_state=torch.zeros(6),
            start_velocity=torch.zeros(6),
            action_mean=torch.zeros(4),
            action_last=torch.zeros(4),
            dt=0.1,
            task_context=torch.zeros(8),
            context_age=0.0,
            timestamp=0.15,
            anchor="short_span",
        )
        self.assertTrue(res_s2["persistent_alert"])

        res_l1 = monitor.evaluate_step(
            start_z=start_z,
            target_z=drift_target,
            start_state=torch.zeros(6),
            start_velocity=torch.zeros(6),
            action_mean=torch.zeros(4),
            action_last=torch.zeros(4),
            dt=0.4,
            task_context=torch.zeros(8),
            context_age=0.0,
            timestamp=0.15,
            anchor="long_span",
        )
        self.assertTrue(res_l1["is_step_drift"])
        self.assertFalse(res_l1["persistent_alert"])


class SentinelCLISmokeTests(unittest.TestCase):
    def test_full_extract_train_calibrate_shadow_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            meta = _make_metadata(shallow_weights_fingerprint="shallow_frozen_model_v1")

            rec_train = _make_recording("ep_train_1")
            rec_val = _make_recording("ep_val_1")
            rec_test = _make_recording("ep_test_1")

            raw_train_file = tmp_path / "raw_train.pt"
            raw_val_file = tmp_path / "raw_val.pt"
            raw_test_file = tmp_path / "raw_test.pt"

            # Recordings payload includes both metadata and recordings
            torch.save({"metadata": meta.to_dict(), "recordings": [rec_train]}, raw_train_file)
            torch.save({"metadata": meta.to_dict(), "recordings": [rec_val]}, raw_val_file)
            torch.save({"metadata": meta.to_dict(), "recordings": [rec_test]}, raw_test_file)

            train_ds_file = tmp_path / "train_ds.pt"
            val_ds_file = tmp_path / "val_ds.pt"
            test_ds_file = tmp_path / "test_ds.pt"

            # 1. CLI extract train
            subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "extract",
                    "--recordings-path",
                    str(raw_train_file),
                    "--output",
                    str(train_ds_file),
                    "--spans",
                    "1",
                    "2",
                ],
                check=True,
            )
            self.assertTrue(train_ds_file.exists())

            # 2. CLI extract val
            subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "extract",
                    "--recordings-path",
                    str(raw_val_file),
                    "--output",
                    str(val_ds_file),
                    "--spans",
                    "1",
                    "2",
                ],
                check=True,
            )
            self.assertTrue(val_ds_file.exists())

            # 2b. CLI extract test (third independent episode)
            subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "extract",
                    "--recordings-path",
                    str(raw_test_file),
                    "--output",
                    str(test_ds_file),
                    "--spans",
                    "1",
                    "2",
                ],
                check=True,
            )
            self.assertTrue(test_ds_file.exists())

            # 3. CLI train
            model_out = tmp_path / "sentinel_model.pt"
            subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "train",
                    "--train-data",
                    str(train_ds_file),
                    "--val-data",
                    str(val_ds_file),
                    "--output",
                    str(model_out),
                    "--epochs",
                    "2",
                    "--batch-size",
                    "8",
                    "--hidden-dim",
                    "32",
                ],
                check=True,
            )
            self.assertTrue(model_out.exists())

            # Verify checkpoint preserved exact metadata including shallow_weights_fingerprint
            ckpt = torch.load(model_out, map_location="cpu", weights_only=False)
            self.assertEqual(
                ckpt["metadata"]["shallow_weights_fingerprint"],
                "shallow_frozen_model_v1",
            )
            self.assertIn("predictor_state_hash", ckpt)
            self.assertIsNotNone(ckpt["predictor_state_hash"])
            self.assertIn("val_loss", ckpt)
            self.assertIsNotNone(ckpt["val_loss"])

            # 4. CLI calibrate
            profile_out = tmp_path / "calibration_profile.json"
            subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "calibrate",
                    "--model-checkpoint",
                    str(model_out),
                    "--val-data",
                    str(val_ds_file),
                    "--output-profile",
                    str(profile_out),
                    "--quantile",
                    "0.90",
                    "--persistence-seconds",
                    "0.4",
                ],
                check=True,
            )
            self.assertTrue(profile_out.exists())

            with open(profile_out) as f:
                prof = json.load(f)
            self.assertEqual(
                prof["metadata"]["shallow_weights_fingerprint"],
                "shallow_frozen_model_v1",
            )
            self.assertEqual(prof["predictor_state_hash"], ckpt["predictor_state_hash"])
            self.assertEqual(prof["calibration_episodes"], ["ep_val_1"])
            self.assertGreater(len(prof["bins"]), 0)

            # 4b. Verify profile old missing list rejection
            old_prof_dict = dict(prof)
            del old_prof_dict["calibration_episodes"]
            with self.assertRaises(ValueError):
                SentinelCalibrationProfile.from_dict(old_prof_dict)

            # 5a. Verify Rejection Class 1: Training episode intersection in held_out shadow
            reject_train_log = tmp_path / "reject_train.jsonl"
            proc_train = subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "shadow",
                    "--dataset",
                    str(train_ds_file),
                    "--model-checkpoint",
                    str(model_out),
                    "--profile",
                    str(profile_out),
                    "--output-log",
                    str(reject_train_log),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc_train.returncode, 0)
            self.assertIn("training episodes in held-out mode", proc_train.stderr)
            self.assertFalse(reject_train_log.exists())

            # 5b. Verify Rejection Class 2: Calibration episode intersection in held_out shadow
            reject_val_log = tmp_path / "reject_val.jsonl"
            proc_val = subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "shadow",
                    "--dataset",
                    str(val_ds_file),
                    "--model-checkpoint",
                    str(model_out),
                    "--profile",
                    str(profile_out),
                    "--output-log",
                    str(reject_val_log),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(proc_val.returncode, 0)
            self.assertIn("calibration episodes in held-out mode", proc_val.stderr)
            self.assertFalse(reject_val_log.exists())

            # 5c. CLI shadow replay on third independent episode (held-out default mode)
            shadow_log_out = tmp_path / "shadow_replay.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "shadow",
                    "--dataset",
                    str(test_ds_file),
                    "--model-checkpoint",
                    str(model_out),
                    "--profile",
                    str(profile_out),
                    "--output-log",
                    str(shadow_log_out),
                ],
                check=True,
            )
            self.assertTrue(shadow_log_out.exists())

            lines = shadow_log_out.read_text().strip().split("\n")
            self.assertGreater(len(lines), 0)
            first_entry = json.loads(lines[0])
            self.assertEqual(first_entry["mode"], "held_out")
            self.assertEqual(first_entry["train_overlap"], [])
            self.assertEqual(first_entry["calibration_overlap"], [])
            self.assertIn("start_time", first_entry)
            self.assertIn("end_time", first_entry)
            self.assertIn("is_step_drift", first_entry)
            self.assertIn("persistent_alert", first_entry)

            # 5d. CLI shadow replay with --allow-data-overlap (diagnostic mode labels)
            diag_log_out = tmp_path / "shadow_diag.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    "train_sentinel.py",
                    "shadow",
                    "--dataset",
                    str(val_ds_file),
                    "--model-checkpoint",
                    str(model_out),
                    "--profile",
                    str(profile_out),
                    "--output-log",
                    str(diag_log_out),
                    "--allow-data-overlap",
                ],
                check=True,
            )
            self.assertTrue(diag_log_out.exists())
            diag_lines = diag_log_out.read_text().strip().split("\n")
            self.assertGreater(len(diag_lines), 0)
            diag_entry = json.loads(diag_lines[0])
            self.assertEqual(diag_entry["mode"], "diagnostic")
            self.assertEqual(diag_entry["train_overlap"], [])
            self.assertEqual(diag_entry["calibration_overlap"], ["ep_val_1"])


if __name__ == "__main__":
    unittest.main()
