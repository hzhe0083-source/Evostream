"""Unit tests for compact memory replay protocol transition and contract verification."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict

import numpy as np
import pytest
import torch

from fabri_moss.compact_protocol import get_compact_protocol_contract
from fabri_moss.memory_protocol import get_memory_protocol_contract
from fabri_moss.tests.test_memory_transition import _make_contracts
from fabri_moss.tests.test_train_native import TinyNamedPolicy
from fabri_moss.train_native import (
    compute_file_sha256,
    create_native_optimizer_and_scheduler,
    get_protocol_contract_and_format,
    load_native_training_checkpoint,
    parse_args,
    save_native_checkpoint,
    transition_native_checkpoint,
    validate_cli_arguments,
)


def _make_protocol_data_contract(base_dc: Dict[str, Any], protocol_contract: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct stream data contract from base contract using native_data SHA256 formula."""
    dc = dict(base_dc)
    dc["stream_protocol"] = protocol_contract
    sorted_json = json.dumps(protocol_contract, sort_keys=True)
    raw = f"{base_dc['data_fingerprint']}:{sorted_json}"
    dc["data_fingerprint"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return dc


@pytest.mark.parametrize("parent_type", ["dense", "memory"])
def test_transition_parent_to_compact_replay_and_resume(parent_type: str, tmp_path: Path):
    """Test transition from dense or memory replay v1.2 to compact_memory_replay_v1.

    Verifies model tensors, AdamW moments/step, scheduler, RNG, lineage, and strict resume.
    """
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=30, warmup_steps=2)

    # Real autograd step with sum of squared parameters
    loss = sum((p ** 2).sum() for p in policy.parameters())
    loss.backward()
    opt.step()
    sched.step()

    assert len(opt.state) > 0
    for p in policy.parameters():
        if p.requires_grad:
            assert p in opt.state
            assert "exp_avg" in opt.state[p]
            assert "exp_avg_sq" in opt.state[p]
            assert opt.state[p]["step"] == 1

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    src_meta, dense_tr_contract, dense_dc, val_dc = _make_contracts()

    if parent_type == "dense":
        parent_tr = copy.deepcopy(dense_tr_contract)
        parent_train_dc = copy.deepcopy(dense_dc)
        parent_val_dc = copy.deepcopy(val_dc)
        expected_from_protocol = "native_multiframe_v1"
    else:
        mem_contract = get_memory_protocol_contract()
        parent_tr = copy.deepcopy(dense_tr_contract)
        parent_tr["format"] = "native_memory_replay_v1"
        parent_tr["stream_protocol"] = mem_contract
        parent_train_dc = _make_protocol_data_contract(dense_dc, mem_contract)
        parent_val_dc = _make_protocol_data_contract(val_dc, mem_contract)
        expected_from_protocol = "native_memory_replay_v1"

    (tmp_path / f"parent_{parent_type}").mkdir()
    parent_path = save_native_checkpoint(
        output_dir=tmp_path / f"parent_{parent_type}",
        filename="parent.pt",
        policy=policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=parent_tr,
        train_data_contract=parent_train_dc,
        val_data_contract=parent_val_dc,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=8,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )
    parent_sha = compute_file_sha256(parent_path)

    compact_contract, compact_format = get_protocol_contract_and_format("compact_memory_replay_v1")
    assert compact_format == "native_compact_memory_replay_v1"

    exp_compact_tr = copy.deepcopy(dense_tr_contract)
    exp_compact_tr["format"] = compact_format
    exp_compact_tr["stream_protocol"] = compact_contract

    target_policy = TinyNamedPolicy()
    target_opt, target_sched = create_native_optimizer_and_scheduler(target_policy, total_steps=30, warmup_steps=2)

    # Perturb RNG before transition
    _ = random.random(), np.random.rand(), torch.randn(5)

    res_info = transition_native_checkpoint(
        checkpoint_path=parent_path,
        policy=target_policy,
        optimizer=target_opt,
        scheduler=target_sched,
        expected_source_metadata=src_meta,
        expected_training_contract=exp_compact_tr,
        expected_train_data_contract=dense_dc,
        expected_val_data_contract=val_dc,
        stream_protocol="compact_memory_replay_v1",
        device=torch.device("cpu"),
    )

    lineage = res_info["stage_lineage"]
    assert lineage is not None
    assert lineage["parent_checkpoint_sha256"] == parent_sha
    assert lineage["from_protocol"] == expected_from_protocol
    assert lineage["to_protocol"] == "compact_memory_replay_v1"
    assert lineage["parent_global_step"] == 1
    assert lineage["parent_epoch"] == 0
    assert lineage["parent_batch_cursor"] == 4

    for k, v in policy.state_dict().items():
        assert torch.equal(target_policy.state_dict()[k], v)

    for target_p, p in zip(target_policy.parameters(), policy.parameters()):
        if p.requires_grad:
            assert target_p in target_opt.state
            assert opt.state[p]["step"] == target_opt.state[target_p]["step"] == 1
            assert torch.equal(opt.state[p]["exp_avg"], target_opt.state[target_p]["exp_avg"])
            assert torch.equal(opt.state[p]["exp_avg_sq"], target_opt.state[target_p]["exp_avg_sq"])

    assert target_sched.last_epoch == sched.last_epoch == 1
    assert res_info["global_step"] == 1
    assert res_info["epoch"] == 0
    assert res_info["batch_cursor"] == 4
    assert res_info["epoch_targets_seen"] == 8

    saved_rng = torch.load(parent_path, map_location="cpu", weights_only=False)["rng_states_per_rank"][0]
    assert random.getstate() == saved_rng["python"]
    current_np = np.random.get_state()
    assert current_np[0] == saved_rng["numpy"][0]
    assert np.array_equal(current_np[1], saved_rng["numpy"][1])
    assert current_np[2:] == saved_rng["numpy"][2:]
    assert torch.equal(torch.get_rng_state(), saved_rng["torch_cpu"])

    # Save new compact stage checkpoint and test exact resume
    compact_run_dir = tmp_path / f"compact_{parent_type}"
    compact_run_dir.mkdir(parents=True, exist_ok=True)
    compact_train_dc = _make_protocol_data_contract(dense_dc, compact_contract)
    compact_val_dc = _make_protocol_data_contract(val_dc, compact_contract)

    target_opt.zero_grad(set_to_none=True)
    sum(p.square().sum() for p in target_policy.parameters()).backward()
    target_opt.step()
    target_sched.step()
    compact_step2_path = save_native_checkpoint(
        output_dir=compact_run_dir,
        filename="compact_step2.pt",
        policy=target_policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=exp_compact_tr,
        train_data_contract=compact_train_dc,
        val_data_contract=compact_val_dc,
        optimizer=target_opt,
        scheduler=target_sched,
        global_step=2,
        epoch=0,
        batch_cursor=8,
        epoch_targets_seen=16,
        source_metadata=dict(src_meta, stage_lineage=lineage),
        device=torch.device("cpu"),
    )

    resume_policy = TinyNamedPolicy()
    resume_opt, resume_sched = create_native_optimizer_and_scheduler(resume_policy, total_steps=30, warmup_steps=2)
    resume_info = load_native_training_checkpoint(
        checkpoint_path=compact_step2_path,
        policy=resume_policy,
        optimizer=resume_opt,
        scheduler=resume_sched,
        expected_source_metadata=dict(src_meta, stage_lineage=lineage),
        expected_training_contract=exp_compact_tr,
        expected_train_data_contract=compact_train_dc,
        expected_val_data_contract=compact_val_dc,
        device=torch.device("cpu"),
        mode="resume",
    )
    assert resume_info["global_step"] == 2
    assert resume_info["stage_lineage"] == lineage

    # Ordinary resume targeting old parent training contract must reject
    with pytest.raises(ValueError, match="Training contract mismatch"):
        load_native_training_checkpoint(
            checkpoint_path=compact_step2_path,
            policy=resume_policy,
            optimizer=resume_opt,
            scheduler=resume_sched,
            expected_source_metadata=dict(src_meta, stage_lineage=lineage),
            expected_training_contract=parent_tr,
            expected_train_data_contract=compact_train_dc,
            expected_val_data_contract=compact_val_dc,
            device=torch.device("cpu"),
            mode="resume",
        )


@pytest.mark.parametrize(
    "mutation_case",
    ["wrong_version", "wrong_data_fingerprint", "geometry_diff", "source_sha_diff"],
)
def test_memory_parent_rejections_preserve_target_weights(mutation_case: str, tmp_path: Path):
    """Corrupted memory parent contracts trigger strict reject before modifying target weights."""
    policy = TinyNamedPolicy()
    opt, sched = create_native_optimizer_and_scheduler(policy, total_steps=30, warmup_steps=2)

    sum(p.square().sum() for p in policy.parameters()).backward()
    opt.step()
    sched.step()
    src_meta, dense_tr_contract, dense_dc, val_dc = _make_contracts()
    mem_contract = get_memory_protocol_contract()

    parent_tr = copy.deepcopy(dense_tr_contract)
    parent_tr["format"] = "native_memory_replay_v1"
    parent_tr["stream_protocol"] = copy.deepcopy(mem_contract)
    parent_train_dc = _make_protocol_data_contract(dense_dc, mem_contract)
    parent_val_dc = _make_protocol_data_contract(val_dc, mem_contract)
    exp_source_meta = copy.deepcopy(src_meta)

    if mutation_case == "wrong_version":
        parent_tr["stream_protocol"]["version"] = "9.9"
    elif mutation_case == "wrong_data_fingerprint":
        parent_train_dc["data_fingerprint"] = "bad_fingerprint_hash"
    elif mutation_case == "geometry_diff":
        parent_train_dc["history_frames"] = 99
    elif mutation_case == "source_sha_diff":
        exp_source_meta["checkpoint_sha256"] = "mismatched_source_sha"

    (tmp_path / f"corrupt_{mutation_case}").mkdir()
    parent_path = save_native_checkpoint(
        output_dir=tmp_path / f"corrupt_{mutation_case}",
        filename="corrupt_parent.pt",
        policy=policy,
        config={"dim": 64},
        norm_stats={"mean": 0.0},
        training_contract=parent_tr,
        train_data_contract=parent_train_dc,
        val_data_contract=parent_val_dc,
        optimizer=opt,
        scheduler=sched,
        global_step=1,
        epoch=0,
        batch_cursor=4,
        epoch_targets_seen=8,
        source_metadata=src_meta,
        device=torch.device("cpu"),
    )

    compact_contract, compact_format = get_protocol_contract_and_format("compact_memory_replay_v1")
    exp_compact_tr = copy.deepcopy(dense_tr_contract)
    exp_compact_tr["format"] = compact_format
    exp_compact_tr["stream_protocol"] = compact_contract

    target_policy = TinyNamedPolicy()
    initial_weights = {k: v.clone() for k, v in target_policy.state_dict().items()}
    target_opt, target_sched = create_native_optimizer_and_scheduler(target_policy, total_steps=30, warmup_steps=2)

    with pytest.raises(ValueError):
        transition_native_checkpoint(
            checkpoint_path=parent_path,
            policy=target_policy,
            optimizer=target_opt,
            scheduler=target_sched,
            expected_source_metadata=exp_source_meta,
            expected_training_contract=exp_compact_tr,
            expected_train_data_contract=dense_dc,
            expected_val_data_contract=val_dc,
            stream_protocol="compact_memory_replay_v1",
            device=torch.device("cpu"),
        )

    # Verify target policy weights remain completely untouched
    for k, v in target_policy.state_dict().items():
        assert torch.equal(v, initial_weights[k])


def test_compact_cli_arguments_and_protocol_contract(tmp_path: Path):
    """Test CLI argument parsing, validation, and contract specification for compact transition."""
    ckpt_path = tmp_path / "toy_dense.pt"
    ckpt_path.touch()

    args = parse_args([
        "--output-dir", str(tmp_path / "new_compact_out"),
        "--stream-protocol", "compact_memory_replay_v1",
        "--transition-from", str(ckpt_path),
    ])
    assert args.stream_protocol == "compact_memory_replay_v1"
    assert args.transition_from == str(ckpt_path)
    validate_cli_arguments(args)

    contract, fmt = get_protocol_contract_and_format("compact_memory_replay_v1")
    assert fmt == "native_compact_memory_replay_v1"
    assert contract["protocol_name"] == "compact_memory_replay_v1"
    assert contract["version"] == "1.0"
    assert "compact_memory_config" in contract
    assert "representation" in contract
