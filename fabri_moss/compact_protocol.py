"""compact_memory_replay_v1 protocol helper and specification for NativeTrainingDataset."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict

from fabri_moss.compact_memory import CompactMemoryConfig
from fabri_moss.memory_protocol import get_memory_protocol_contract


COMPACT_MEMORY_REPLAY_V1_CONFIG: Dict[str, Any] = {
    "protocol_name": "compact_memory_replay_v1",
    "version": "1.0",
    "representation": {
        "intermediate_grid": 8,
        "protected_full_decision_frames": True,
        "full_current_tokens": 1024,
    },
    "temporal": {
        "temporal_rope": True,
        "strength": 1.0,
        "time_unit_seconds": 1.0,
        "rotary_fraction": 0.25,
    },
    "lifecycle": {
        "postquery_consolidation": True,
        "nextquery_effective": True,
        "summary_timestamp": "end_time",
        "summary_text_preserves_range_and_count": True,
    },
}


def get_compact_protocol_contract() -> Dict[str, Any]:
    """Return JSON-serializable stable contract specification for compact_memory_replay_v1."""
    default_config = CompactMemoryConfig()
    config_dict = dataclasses.asdict(default_config)

    contract: Dict[str, Any] = {
        "protocol_name": COMPACT_MEMORY_REPLAY_V1_CONFIG["protocol_name"],
        "version": COMPACT_MEMORY_REPLAY_V1_CONFIG["version"],
        "compact_memory_config": config_dict,
        "representation": dict(COMPACT_MEMORY_REPLAY_V1_CONFIG["representation"]),
        "temporal": dict(COMPACT_MEMORY_REPLAY_V1_CONFIG["temporal"]),
        "lifecycle": dict(COMPACT_MEMORY_REPLAY_V1_CONFIG["lifecycle"]),
        "base_sampling_contract": get_memory_protocol_contract(),
    }
    return copy.deepcopy(contract)
