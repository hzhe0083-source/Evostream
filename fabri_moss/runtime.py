"""Runtime loader for native FabriVLA checkpoints and isolated module loading.

Ensures no conflicts with repo-root model.py or other packages. Sets offline
environment variables and loads native modules via an isolated namespace.
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type


def compute_file_sha256(filepath: str | Path, block_size: int = 65536) -> str:
    """Compute sha256 checksum of a file for strict provenance verification."""
    sha = hashlib.sha256()
    with open(str(filepath), "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            sha.update(block)
    return sha.hexdigest()


def import_native_fabrivla_modules(fabri_root: str | Path) -> Tuple[Type[Any], Type[Any], Type[Any]]:
    """Import native InternVL3Embedder, FabriVLAFlowPolicy, and ActionHeadConfig in an isolated package alias.

    Prevents conflicting with any root `model.py` in the working directory.
    Uses importlib to load modules into `_fabri_native` and `_fabri_native.model`.
    Strictly verifies that existing alias modules match the requested fabri_root, raising on mismatch.

    Returns:
        (InternVL3Embedder, FabriVLAFlowPolicy, ActionHeadConfig)
    """
    fabri_root = Path(fabri_root).resolve()
    src_dir = fabri_root / "src"
    model_dir = src_dir / "model"

    if not model_dir.exists():
        if (fabri_root / "fabri-vla" / "src" / "model").exists():
            fabri_root = fabri_root / "fabri-vla"
            src_dir = fabri_root / "src"
            model_dir = src_dir / "model"
        else:
            raise FileNotFoundError(f"Native model directory not found at {model_dir}")

    pkg_name = "_fabri_native"
    sub_name = f"{pkg_name}.model"

    # Strict path check if already imported
    if pkg_name in sys.modules:
        existing_pkg = sys.modules[pkg_name]
        existing_locs = getattr(existing_pkg, "__path__", [])
        if existing_locs and Path(existing_locs[0]).resolve() != src_dir:
            raise RuntimeError(
                f"Isolated namespace '{pkg_name}' already loaded from {existing_locs[0]}, "
                f"which does not match requested path {src_dir}"
            )
    else:
        spec = importlib.machinery.ModuleSpec(pkg_name, None, is_package=True)
        spec.submodule_search_locations = [str(src_dir)]
        pkg_mod = importlib.util.module_from_spec(spec)
        sys.modules[pkg_name] = pkg_mod

    if sub_name in sys.modules:
        existing_sub = sys.modules[sub_name]
        existing_sub_locs = getattr(existing_sub, "__path__", [])
        if existing_sub_locs and Path(existing_sub_locs[0]).resolve() != model_dir:
            raise RuntimeError(
                f"Isolated namespace '{sub_name}' already loaded from {existing_sub_locs[0]}, "
                f"which does not match requested path {model_dir}"
            )
    else:
        sub_spec = importlib.machinery.ModuleSpec(sub_name, None, is_package=True)
        sub_spec.submodule_search_locations = [str(model_dir)]
        sub_mod = importlib.util.module_from_spec(sub_spec)
        sys.modules[sub_name] = sub_mod
        setattr(sys.modules[pkg_name], "model", sub_mod)

    # Load internvl_embedder
    embedder_path = model_dir / "internvl_embedder.py"
    embedder_mod_name = f"{sub_name}.internvl_embedder"
    if embedder_mod_name not in sys.modules:
        e_spec = importlib.util.spec_from_file_location(embedder_mod_name, str(embedder_path))
        if e_spec is None or e_spec.loader is None:
            raise ImportError(f"Cannot create spec for {embedder_path}")
        e_mod = importlib.util.module_from_spec(e_spec)
        sys.modules[embedder_mod_name] = e_mod
        setattr(sys.modules[sub_name], "internvl_embedder", e_mod)
        e_spec.loader.exec_module(e_mod)
    else:
        e_mod = sys.modules[embedder_mod_name]

    # Load action_head
    action_head_path = model_dir / "action_head.py"
    action_head_mod_name = f"{sub_name}.action_head"
    if action_head_mod_name not in sys.modules:
        a_spec = importlib.util.spec_from_file_location(action_head_mod_name, str(action_head_path))
        if a_spec is None or a_spec.loader is None:
            raise ImportError(f"Cannot create spec for {action_head_path}")
        a_mod = importlib.util.module_from_spec(a_spec)
        sys.modules[action_head_mod_name] = a_mod
        setattr(sys.modules[sub_name], "action_head", a_mod)
        a_spec.loader.exec_module(a_mod)
    else:
        a_mod = sys.modules[action_head_mod_name]

    InternVL3Embedder: Type[Any] = getattr(e_mod, "InternVL3Embedder")
    FabriVLAFlowPolicy: Type[Any] = getattr(a_mod, "FabriVLAFlowPolicy")
    ActionHeadConfig: Type[Any] = getattr(a_mod, "ActionHeadConfig")

    return InternVL3Embedder, FabriVLAFlowPolicy, ActionHeadConfig


def select_norm_stats(
    raw_stats: Dict[str, Any],
    arm_key: str = "metaworld_sawyer",
) -> Dict[str, Any]:
    """Select normalization stats for MetaWorld Sawyer or fallback to top-level stats."""
    if not isinstance(raw_stats, dict):
        raise ValueError(f"raw_stats must be a dict, got {type(raw_stats)}")

    if arm_key in raw_stats:
        return raw_stats[arm_key]
    if "arm2stats_dict" in raw_stats and arm_key in raw_stats["arm2stats_dict"]:
        return raw_stats["arm2stats_dict"][arm_key]
    if "observation.state" in raw_stats and "action" in raw_stats:
        return raw_stats

    raise KeyError(f"Could not find valid norm_stats for {arm_key} in keys: {list(raw_stats.keys())}")


def load_native_checkpoint(
    fabri_root: str | Path = "/root/FabriVLA",
    checkpoint_path: str | Path = "/root/models/FabriVLA/checkpoint_step_93000.pt",
    vlm_path: str | Path = "/root/models/InternVL3_5-1B",
    device: str = "cpu",
    arm_key: str = "metaworld_sawyer",
    trainable: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Strictly load native FabriVLA checkpoint from disk.

    Args:
        trainable: If True, casts constructed policy to FP32 before loading state dict
            so FP32 source parameters are not rounded to BF16, and enables requires_grad=True
            on all parameters.

    Returns:
        policy: FabriVLAFlowPolicy instantiated with native weights
        checkpoint_config: Clean config dict extracted from checkpoint
        norm_stats: Selected normalization stats for the specified arm
        metadata: Diagnostic provenance metadata including sha256 checksum
    """
    import torch

    checkpoint_path = Path(checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    vlm_path = Path(vlm_path).resolve()
    fabri_root = Path(fabri_root).resolve()

    # Set offline variables before loading native models
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_DISABLED"] = "true"

    # Compute sha256 checksum of checkpoint for provenance
    ckpt_sha256 = compute_file_sha256(checkpoint_path)

    InternVL3Embedder, FabriVLAFlowPolicy, ActionHeadConfig = import_native_fabrivla_modules(fabri_root)

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False, mmap=True)

    raw_config = ckpt.get("config", {})
    if not isinstance(raw_config, dict):
        if hasattr(raw_config, "__dict__"):
            raw_config = vars(raw_config)
        else:
            raw_config = {}

    head_fields = {f.name for f in dataclasses.fields(ActionHeadConfig)}
    head_config_dict = {k: v for k, v in raw_config.items() if k in head_fields}

    if "shallow_fusion" in raw_config:
        head_config_dict["shallow_fusion"] = raw_config["shallow_fusion"]
    if "shallow_layer_index" in raw_config:
        head_config_dict["shallow_layer_index"] = raw_config["shallow_layer_index"]

    action_head_config = ActionHeadConfig(**head_config_dict)

    raw_norm_stats = ckpt.get("norm_stats", {})
    norm_stats = select_norm_stats(raw_norm_stats, arm_key=arm_key)

    embedder = InternVL3Embedder(
        model_name=str(vlm_path),
        image_size=raw_config.get("image_size", 448),
        device=device,
        num_keep_layers=raw_config.get("num_keep_layers", 14),
    )

    policy = FabriVLAFlowPolicy(
        config=action_head_config,
        embedder=embedder,
        freeze_embedder=not trainable,
    )

    if trainable:
        policy.float()

    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    # Only strip 'module.' prefix if ALL keys start with 'module.' (standard DDP saving)
    # Do not silently strip arbitrary prefixes
    if all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Strict load into policy
    incompatible = policy.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict checkpoint loading failed! Missing keys: {incompatible.missing_keys}, "
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    policy.to(device)
    policy.action_head.float()

    if trainable:
        for p in policy.parameters():
            p.requires_grad = True

    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_sha256,
        "vlm_path": str(vlm_path),
        "fabri_root": str(fabri_root),
        "device": device,
        "step": ckpt.get("step", ckpt.get("global_step", None)),
        "head_config": head_config_dict,
        "arm_key": arm_key,
        "total_state_dict_keys": len(state_dict),
        "trainable": trainable,
        "precision": "float32" if trainable else "mixed",
    }

    # Record actual attention backend in metadata if accessible, without forcing FA2 on CPU
    try:
        if hasattr(policy, "embedder") and hasattr(policy.embedder, "model"):
            m = policy.embedder.model
            lm_attn = getattr(getattr(getattr(m, "language_model", None), "config", None), "_attn_implementation", None)
            v_layers = getattr(getattr(getattr(m, "vision_model", None), "encoder", None), "layers", None)
            v_fa2 = all(getattr(getattr(l, "attn", None), "use_flash_attn", False) for l in v_layers) if v_layers else False
            metadata["actual_backend"] = {
                "language_attn_implementation": lm_attn,
                "vision_layers_count": len(v_layers) if v_layers else 0,
                "vision_use_flash_attn": v_fa2,
            }
    except Exception:
        pass

    return policy, raw_config, norm_stats, metadata


def assert_native_fa2(policy: Any) -> Dict[str, Any]:
    """Strictly assert that policy uses native FlashAttention-2 for both LM and Vision.

    Checks:
    - language_model.config._attn_implementation == 'flash_attention_2'
    - vision_model.encoder.layers has 24 layers and all layers have attn.use_flash_attn == True
    - language_model.model.layers has 14 layers

    Returns diagnostic dict with LM backend, vision layer count, enabled status for training/eval provenance.
    Raises RuntimeError on any mismatch.
    """
    if not hasattr(policy, "embedder") or not hasattr(policy.embedder, "model"):
        raise RuntimeError("Policy does not have embedder.model structure")

    model = policy.embedder.model

    # Check language model
    if not hasattr(model, "language_model") or not hasattr(model.language_model, "config"):
        raise RuntimeError("Policy embedder.model missing language_model or language_model.config")

    lm_config = model.language_model.config
    attn_impl = getattr(lm_config, "_attn_implementation", None)
    if attn_impl != "flash_attention_2":
        raise RuntimeError(
            f"Language model _attn_implementation is '{attn_impl}', expected 'flash_attention_2'"
        )

    # Check vision model
    if not hasattr(model, "vision_model") or not hasattr(model.vision_model, "encoder") or not hasattr(model.vision_model.encoder, "layers"):
        raise RuntimeError("Policy embedder.model missing vision_model.encoder.layers")

    v_layers = model.vision_model.encoder.layers
    if len(v_layers) != 24:
        raise RuntimeError(f"Vision model encoder has {len(v_layers)} layers, expected 24")

    bad_v_layers = [i for i, l in enumerate(v_layers) if not getattr(getattr(l, "attn", None), "use_flash_attn", False)]
    if bad_v_layers:
        raise RuntimeError(
            f"Vision model encoder layers {bad_v_layers} do not have attn.use_flash_attn == True"
        )

    # Check language layers count
    l_layers = getattr(getattr(getattr(model, "language_model", None), "model", None), "layers", None)
    if l_layers is None or len(l_layers) != 14:
        count = len(l_layers) if l_layers is not None else 0
        raise RuntimeError(f"Language model has {count} layers, expected 14")

    diag = {
        "language_attn_implementation": attn_impl,
        "language_layers_count": len(l_layers),
        "vision_layers_count": len(v_layers),
        "vision_flash_attn": True,
        "native_fa2_enabled": True,
    }
    return diag


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FabriVLA Runtime Native Loader")
    parser.add_argument("--fabri-root", type=str, default="/root/FabriVLA", help="Path to native FabriVLA repo")
    parser.add_argument("--checkpoint", type=str, default="/root/models/FabriVLA/checkpoint_step_93000.pt", help="Path to checkpoint .pt")
    parser.add_argument("--vlm", type=str, default="/root/models/InternVL3_5-1B", help="Path to local InternVL model")
    parser.add_argument("--device", type=str, default="cpu", help="Target device (cpu/cuda)")
    parser.add_argument("--arm-key", type=str, default="metaworld_sawyer", help="Arm key for norm stats")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    policy, config, norm_stats, meta = load_native_checkpoint(
        fabri_root=args.fabri_root,
        checkpoint_path=args.checkpoint,
        vlm_path=args.vlm,
        device=args.device,
        arm_key=args.arm_key,
    )
    print(f"[runtime] Successfully initialized native policy on {args.device}. Metadata: {meta}")
