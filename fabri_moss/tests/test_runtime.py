"""Unit tests for fabri_moss.runtime assert_native_fa2 and metadata."""

from __future__ import annotations

import types
import pytest
import torch
import torch.nn as nn

from fabri_moss.runtime import assert_native_fa2


def make_mock_fa2_policy(
    attn_impl: str = "flash_attention_2",
    num_v_layers: int = 24,
    v_fa2: bool = True,
    num_l_layers: int = 14,
):
    policy = types.SimpleNamespace()
    embedder = types.SimpleNamespace()
    model = types.SimpleNamespace()

    # Language model
    lm_config = types.SimpleNamespace(_attn_implementation=attn_impl)
    l_layers = [types.SimpleNamespace() for _ in range(num_l_layers)]
    lm_model = types.SimpleNamespace(layers=l_layers)
    language_model = types.SimpleNamespace(config=lm_config, model=lm_model)

    # Vision model
    v_layers = []
    for _ in range(num_v_layers):
        layer = types.SimpleNamespace(attn=types.SimpleNamespace(use_flash_attn=v_fa2))
        v_layers.append(layer)
    encoder = types.SimpleNamespace(layers=v_layers)
    vision_model = types.SimpleNamespace(encoder=encoder)

    model.language_model = language_model
    model.vision_model = vision_model
    embedder.model = model
    policy.embedder = embedder

    return policy


def test_assert_native_fa2_success():
    policy = make_mock_fa2_policy()
    diag = assert_native_fa2(policy)
    assert diag["language_attn_implementation"] == "flash_attention_2"
    assert diag["language_layers_count"] == 14
    assert diag["vision_layers_count"] == 24
    assert diag["vision_flash_attn"] is True
    assert diag["native_fa2_enabled"] is True


def test_assert_native_fa2_rejects_eager():
    policy = make_mock_fa2_policy(attn_impl="eager")
    with pytest.raises(RuntimeError, match="Language model _attn_implementation is 'eager'"):
        assert_native_fa2(policy)


def test_assert_native_fa2_rejects_sdpa():
    policy = make_mock_fa2_policy(attn_impl="sdpa")
    with pytest.raises(RuntimeError, match="expected 'flash_attention_2'"):
        assert_native_fa2(policy)


def test_assert_native_fa2_rejects_vision_without_fa2():
    policy = make_mock_fa2_policy(v_fa2=False)
    with pytest.raises(RuntimeError, match="do not have attn.use_flash_attn == True"):
        assert_native_fa2(policy)


def test_assert_native_fa2_rejects_wrong_vision_layers():
    policy = make_mock_fa2_policy(num_v_layers=12)
    with pytest.raises(RuntimeError, match="encoder has 12 layers, expected 24"):
        assert_native_fa2(policy)


def test_assert_native_fa2_rejects_wrong_language_layers():
    policy = make_mock_fa2_policy(num_l_layers=12)
    with pytest.raises(RuntimeError, match="Language model has 12 layers, expected 14"):
        assert_native_fa2(policy)
