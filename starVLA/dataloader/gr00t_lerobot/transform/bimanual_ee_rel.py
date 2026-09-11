"""Portable bimanual SE(3) preprocessing for canonical LeRobot datasets.

This module intentionally has no dependency on vla-hub.  The row-major rot6d
and SE(3) routines mirror ``vlahub.geometry``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from pydantic import Field, PrivateAttr

from ..schema import DatasetMetadata, DatasetStatisticalValues
from .base import InvertibleModalityTransform


ACTION_DIM = 20
STATE_DIM = 31
ACTION_ROT6D_DIMS = np.r_[3:9, 13:19]
STATE_ROT6D_DIMS = np.r_[3:9, 13:19, 23:29]
ACTION_NORMALIZED_DIMS = np.setdiff1d(np.arange(ACTION_DIM), ACTION_ROT6D_DIMS)
STATE_NORMALIZED_DIMS = np.setdiff1d(np.arange(STATE_DIM), STATE_ROT6D_DIMS)


def rot6d_from_mat(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    return rotation[..., :2, :].reshape(rotation.shape[:-2] + (6,)).copy()


def mat_from_rot6d(rot6d: np.ndarray) -> np.ndarray:
    rot6d = np.asarray(rot6d, dtype=np.float64)
    first, second = rot6d[..., :3], rot6d[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first_unit = first / np.maximum(first_norm, 1e-8)
    second_orthogonal = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    if np.any(first_norm[..., 0] < 1e-8) or np.any(second_norm[..., 0] < 1e-8):
        raise ValueError("degenerate rot6d input")
    second_unit = second_orthogonal / second_norm
    return np.stack([first_unit, second_unit, np.cross(first_unit, second_unit)], axis=-2)


def pose9_to_mat(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    result = np.zeros(pose.shape[:-1] + (4, 4), dtype=np.float64)
    result[..., :3, :3] = mat_from_rot6d(pose[..., 3:9])
    result[..., :3, 3] = pose[..., :3]
    result[..., 3, 3] = 1.0
    return result


def mat_to_pose9(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    return np.concatenate(
        [transform[..., :3, 3], rot6d_from_mat(transform[..., :3, :3])], axis=-1
    )


def se3_inv(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64)
    rotation_t = np.swapaxes(transform[..., :3, :3], -1, -2)
    result = np.zeros_like(transform)
    result[..., :3, :3] = rotation_t
    result[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", rotation_t, transform[..., :3, 3]
    )
    result[..., 3, 3] = 1.0
    return result


def _arm_pose(values: np.ndarray, offset: int) -> np.ndarray:
    return pose9_to_mat(
        np.concatenate([values[..., offset : offset + 3], values[..., offset + 3 : offset + 9]], axis=-1)
    )


def relativize_action_chunk(state_current: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Absolute canonical (H,20) actions -> current-anchor-relative actions."""
    state_current = np.asarray(state_current, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if state_current.shape != (ACTION_DIM,) or actions.shape != (24, ACTION_DIM):
        raise ValueError(f"expected state (20,) and actions (24,20), got {state_current.shape}, {actions.shape}")
    output = actions.copy()
    for offset in (0, 10):
        relative = se3_inv(_arm_pose(state_current, offset)) @ _arm_pose(actions, offset)
        pose = mat_to_pose9(relative)
        output[..., offset : offset + 3] = pose[..., :3]
        output[..., offset + 3 : offset + 9] = pose[..., 3:9]
    _require_finite("relative action", output)
    return output


def augment_rel_ee_history(state_past: np.ndarray, state_current: np.ndarray) -> np.ndarray:
    """Construct the 31-D rel_ee_history state with stride handled by sampling."""
    past = np.asarray(state_past, dtype=np.float64)
    current = np.asarray(state_current, dtype=np.float64)
    if past.shape != (ACTION_DIM,) or current.shape != (ACTION_DIM,):
        raise ValueError(f"expected two (20,) states, got {past.shape}, {current.shape}")
    shell = current.copy()
    for offset in (0, 10):
        pose = mat_to_pose9(se3_inv(_arm_pose(current, offset)) @ _arm_pose(past, offset))
        shell[offset : offset + 3] = pose[:3]
        shell[offset + 3 : offset + 9] = pose[3:9]
    inter_gripper = mat_to_pose9(se3_inv(_arm_pose(current, 0)) @ _arm_pose(current, 10))
    output = np.concatenate([shell, inter_gripper, past[[9, 19]]])[None, :]
    if output.shape != (1, STATE_DIM):
        raise AssertionError(f"invalid model state shape {output.shape}")
    _require_finite("model state", output)
    return output


def _require_finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _extract_stats(document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = document.get("norm_stats", document)
    if "state" in payload and ("actions" in payload or "action" in payload):
        return payload["state"], payload.get("actions", payload.get("action"))
    if len(payload) == 1:
        nested = next(iter(payload.values()))
        if isinstance(nested, dict) and "state" in nested and "action" in nested:
            return nested["state"], nested["action"]
    raise ValueError("stats JSON must contain state and actions/action statistics")


def load_bimanual_statistics(path: str | Path) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    stats_path = Path(path).expanduser()
    if stats_path.is_dir():
        stats_path = stats_path / "norm_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(f"bimanual statistics not found: {stats_path}")
    state, action = _extract_stats(json.loads(stats_path.read_text()))

    def validate(values: dict[str, Any], dim: int, label: str) -> dict[str, list[float]]:
        output = {}
        for key in ("mean", "std", "q01", "q99"):
            array = np.asarray(values[key], dtype=np.float64)
            if array.shape != (dim,):
                raise ValueError(f"{label}.{key} must have shape ({dim},), got {array.shape}")
            _require_finite(f"{label}.{key}", array)
            output[key] = array.tolist()
        q01, q99 = np.asarray(output["q01"]), np.asarray(output["q99"])
        if np.any(q01 > q99):
            raise ValueError(f"{label} has q01 > q99")
        for key, fallback in (("min", q01), ("max", q99)):
            array = np.asarray(values.get(key, fallback), dtype=np.float64)
            if array.shape != (dim,):
                raise ValueError(f"{label}.{key} must have shape ({dim},), got {array.shape}")
            _require_finite(f"{label}.{key}", array)
            output[key] = array.tolist()
        return output

    return validate(state, STATE_DIM, "state"), validate(action, ACTION_DIM, "actions")


def _normalize_q01_q99(values: np.ndarray, stats: dict[str, list[float]], dims: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    low = np.asarray(stats["q01"], dtype=np.float64)[dims]
    high = np.asarray(stats["q99"], dtype=np.float64)[dims]
    span = high - low
    nonzero = span != 0
    selected = output[..., dims]
    normalized = selected.copy()
    normalized[..., nonzero] = 2.0 * (selected[..., nonzero] - low[nonzero]) / span[nonzero] - 1.0
    normalized[..., ~nonzero] = selected[..., ~nonzero]
    output[..., dims] = np.clip(normalized, -1.0, 1.0)
    return output


def _unnormalize_q01_q99(values: np.ndarray, stats: dict[str, list[float]], dims: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=np.float64).copy()
    low = np.asarray(stats["q01"], dtype=np.float64)[dims]
    high = np.asarray(stats["q99"], dtype=np.float64)[dims]
    output[..., dims] = (output[..., dims] + 1.0) * 0.5 * (high - low) + low
    return output


def inverse_bimanual_actions(
    normalized_relative_actions: np.ndarray,
    absolute_state: np.ndarray,
    statistics_path: str | Path,
) -> np.ndarray:
    """Serving utility: normalized relative chunk -> absolute canonical commands."""
    actions = np.asarray(normalized_relative_actions, dtype=np.float64)
    state = np.asarray(absolute_state, dtype=np.float64)
    if actions.shape[-2:] != (24, ACTION_DIM) or state.shape[-1:] != (ACTION_DIM,):
        raise ValueError(f"expected actions (...,24,20), state (...,20), got {actions.shape}, {state.shape}")
    _, stats = load_bimanual_statistics(statistics_path)
    relative = _unnormalize_q01_q99(actions, stats, ACTION_NORMALIZED_DIMS)
    output = relative.copy()
    for offset in (0, 10):
        absolute = _arm_pose(state, offset)[..., None, :, :] @ _arm_pose(relative, offset)
        pose = mat_to_pose9(absolute)
        output[..., offset : offset + 3] = pose[..., :3]
        output[..., offset + 3 : offset + 9] = pose[..., 3:9]
    _require_finite("absolute action", output)
    return output


class BimanualEERelTransform(InvertibleModalityTransform):
    """Create and selectively normalize portable 20-D actions and 31-D state."""

    apply_to: list[str] = Field(default_factory=lambda: ["state.bimanual", "action.bimanual"])
    statistics_path: str
    _state_stats: dict[str, list[float]] = PrivateAttr(default_factory=dict)
    _action_stats: dict[str, list[float]] = PrivateAttr(default_factory=dict)
    _STATE_KEY: ClassVar[str] = "state.bimanual"
    _ACTION_KEY: ClassVar[str] = "action.bimanual"

    def set_metadata(self, dataset_metadata: DatasetMetadata, original_metadata: DatasetMetadata | None = None):
        self._state_stats, self._action_stats = load_bimanual_statistics(self.statistics_path)
        metadata = dataset_metadata.model_copy(deep=True)
        metadata.statistics.state["bimanual"] = DatasetStatisticalValues.model_validate(
            self._state_stats
        )
        metadata.statistics.action["bimanual"] = DatasetStatisticalValues.model_validate(
            self._action_stats
        )
        metadata.modalities.state["bimanual"].shape = (STATE_DIM,)
        metadata.modalities.state["bimanual"].absolute = False
        metadata.modalities.action["bimanual"].shape = (ACTION_DIM,)
        metadata.modalities.action["bimanual"].absolute = False
        self.dataset_metadata = metadata

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self._state_stats or not self._action_stats:
            raise RuntimeError("transform metadata/statistics were not initialized")
        states = np.asarray(data[self._STATE_KEY])
        actions = np.asarray(data[self._ACTION_KEY])
        if states.shape != (2, ACTION_DIM):
            raise ValueError(f"state.bimanual must be sampled at [-5,0] with shape (2,20), got {states.shape}")
        if actions.shape != (24, ACTION_DIM):
            raise ValueError(f"action.bimanual must have shape (24,20), got {actions.shape}")
        relative_actions = relativize_action_chunk(states[-1], actions)
        model_state = augment_rel_ee_history(states[0], states[-1])
        normalized_actions = _normalize_q01_q99(relative_actions, self._action_stats, ACTION_NORMALIZED_DIMS)
        normalized_state = _normalize_q01_q99(model_state, self._state_stats, STATE_NORMALIZED_DIMS)
        _require_finite("normalized action", normalized_actions)
        _require_finite("normalized state", normalized_state)
        data["action"] = torch.from_numpy(normalized_actions.astype(np.float32))
        data["state"] = torch.from_numpy(normalized_state.astype(np.float32))
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        if "action" in data:
            action = data["action"]
            as_numpy = action.detach().cpu().numpy() if isinstance(action, torch.Tensor) else np.asarray(action)
            data["action"] = _unnormalize_q01_q99(as_numpy, self._action_stats, ACTION_NORMALIZED_DIMS)
        if "state" in data:
            state = data["state"]
            as_numpy = state.detach().cpu().numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
            data["state"] = _unnormalize_q01_q99(as_numpy, self._state_stats, STATE_NORMALIZED_DIMS)
        return data
