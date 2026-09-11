import importlib.util
import json
from pathlib import Path

import numpy as np

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import DatasetMetadata
from starVLA.dataloader.gr00t_lerobot.transform.bimanual_ee_rel import (
    ACTION_ROT6D_DIMS,
    STATE_ROT6D_DIMS,
    BimanualEERelTransform,
    augment_rel_ee_history,
    inverse_bimanual_actions,
    load_bimanual_statistics,
    relativize_action_chunk,
)


REFERENCE_GEOMETRY = Path("/mnt/localssd/lingfeng/vla-hub/vlahub/geometry.py")


def _reference_geometry():
    spec = importlib.util.spec_from_file_location("vlahub_reference_geometry", REFERENCE_GEOMETRY)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _canonical(left_xyz, left_rotation, left_grip, right_xyz, right_rotation, right_grip):
    geometry = _reference_geometry()
    return np.concatenate(
        [
            left_xyz,
            geometry.rot6d_from_mat(left_rotation),
            [left_grip],
            right_xyz,
            geometry.rot6d_from_mat(right_rotation),
            [right_grip],
        ]
    )


def _stats_file(tmp_path):
    action_low, action_high = np.full(20, -10.0), np.full(20, 10.0)
    state_low, state_high = np.full(31, -10.0), np.full(31, 10.0)
    # Deliberately unusable rotation quantiles: rotation slots must be identity-normalized.
    action_low[ACTION_ROT6D_DIMS], action_high[ACTION_ROT6D_DIMS] = 100.0, 200.0
    state_low[STATE_ROT6D_DIMS], state_high[STATE_ROT6D_DIMS] = 100.0, 200.0

    def stats(low, high):
        return {
            "mean": np.zeros_like(low).tolist(),
            "std": np.ones_like(low).tolist(),
            "q01": low.tolist(),
            "q99": high.tolist(),
        }

    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps({"norm_stats": {"state": stats(state_low, state_high), "actions": stats(action_low, action_high)}}))
    return path


def _raw_metadata(stats_path):
    state_stats, action_stats = load_bimanual_statistics(stats_path)
    return DatasetMetadata(
        statistics={
            "state": {"bimanual": {key: value[:20] for key, value in state_stats.items()}},
            "action": {"bimanual": action_stats},
        },
        modalities={
            "video": {},
            "state": {
                "bimanual": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [20],
                    "continuous": True,
                }
            },
            "action": {
                "bimanual": {
                    "absolute": True,
                    "rotation_type": None,
                    "shape": [20],
                    "continuous": True,
                }
            },
        },
        embodiment_tag=EmbodimentTag.PORTABLE_BIMANUAL,
    )


def test_geometry_matches_vlahub_reference():
    geometry = _reference_geometry()
    current = _canonical(
        np.array([0.2, -0.1, 0.5]), _rotation_z(0.3), 0.04,
        np.array([-0.4, 0.3, 0.7]), _rotation_z(-0.2), 0.08,
    )
    past = _canonical(
        np.array([0.1, -0.2, 0.45]), _rotation_z(0.1), 0.03,
        np.array([-0.5, 0.25, 0.65]), _rotation_z(-0.4), 0.07,
    )
    actions = np.stack(
        [
            _canonical(
                np.array([0.2 + 0.01 * h, -0.1, 0.5]), _rotation_z(0.3 + 0.01 * h), 0.04 + h * 0.001,
                np.array([-0.4, 0.3 + 0.005 * h, 0.7]), _rotation_z(-0.2 - 0.01 * h), 0.08,
            )
            for h in range(24)
        ]
    )

    relative = relativize_action_chunk(current, actions)
    expected = actions.copy()
    for offset in (0, 10):
        anchor = geometry.pose9_to_mat(np.r_[current[offset:offset + 3], current[offset + 3:offset + 9]])
        poses = np.c_[actions[:, offset:offset + 3], actions[:, offset + 3:offset + 9]]
        rel_pose = geometry.mat_to_pose9(geometry.relativize(anchor, geometry.pose9_to_mat(poses)))
        expected[:, offset:offset + 3] = rel_pose[:, :3]
        expected[:, offset + 3:offset + 9] = rel_pose[:, 3:9]
    np.testing.assert_allclose(relative, expected, atol=1e-12)

    model_state = augment_rel_ee_history(past, current)
    expected_shell = current.copy()
    for offset in (0, 10):
        cur = geometry.pose9_to_mat(np.r_[current[offset:offset + 3], current[offset + 3:offset + 9]])
        old = geometry.pose9_to_mat(np.r_[past[offset:offset + 3], past[offset + 3:offset + 9]])
        pose = geometry.mat_to_pose9(geometry.relativize(cur, old))
        expected_shell[offset:offset + 3], expected_shell[offset + 3:offset + 9] = pose[:3], pose[3:9]
    left = geometry.pose9_to_mat(np.r_[current[:3], current[3:9]])
    right = geometry.pose9_to_mat(np.r_[current[10:13], current[13:19]])
    inter = geometry.mat_to_pose9(geometry.relativize(left, right))
    expected_state = np.r_[expected_shell, inter, past[[9, 19]]][None]
    np.testing.assert_allclose(model_state, expected_state, atol=1e-12)


def test_transform_shapes_rotation_identity_and_inverse(tmp_path):
    stats_path = _stats_file(tmp_path)
    current = _canonical(
        np.array([0.2, -0.1, 0.5]), _rotation_z(0.3), 0.04,
        np.array([-0.4, 0.3, 0.7]), _rotation_z(-0.2), 0.08,
    )
    past = current.copy()
    actions = np.stack(
        [
            _canonical(
                np.array([0.2 + 0.01 * h, -0.1, 0.5]), _rotation_z(0.3 + 0.01 * h), 0.04,
                np.array([-0.4, 0.3 + 0.005 * h, 0.7]), _rotation_z(-0.2), 0.08,
            )
            for h in range(24)
        ]
    )
    transform = BimanualEERelTransform(statistics_path=str(stats_path))
    transform.set_metadata(_raw_metadata(stats_path))
    assert transform.dataset_metadata.modalities.state["bimanual"].shape == (31,)
    assert len(transform.dataset_metadata.statistics.state["bimanual"].q01) == 31
    output = transform({"state.bimanual": np.stack([past, current]), "action.bimanual": actions.copy()})

    assert tuple(output["action"].shape) == (24, 20)
    assert tuple(output["state"].shape) == (1, 31)
    relative = relativize_action_chunk(current, actions)
    np.testing.assert_allclose(output["action"].numpy()[:, ACTION_ROT6D_DIMS], relative[:, ACTION_ROT6D_DIMS], atol=1e-6)
    model_state = augment_rel_ee_history(past, current)
    np.testing.assert_allclose(output["state"].numpy()[:, STATE_ROT6D_DIMS], model_state[:, STATE_ROT6D_DIMS], atol=1e-6)

    absolute = inverse_bimanual_actions(output["action"].numpy(), current, stats_path)
    np.testing.assert_allclose(absolute, actions, atol=2e-6)


def test_fail_closed_on_wrong_history_or_nonfinite(tmp_path):
    stats_path = _stats_file(tmp_path)
    transform = BimanualEERelTransform(statistics_path=str(stats_path))
    transform.set_metadata(_raw_metadata(stats_path))
    identity = _canonical(np.zeros(3), np.eye(3), 0.0, np.zeros(3), np.eye(3), 0.0)
    actions = np.repeat(identity[None], 24, axis=0)

    import pytest
    with pytest.raises(ValueError, match=r"sampled at \[-5,0\]"):
        transform({"state.bimanual": identity[None], "action.bimanual": actions})
    actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        transform({"state.bimanual": np.stack([identity, identity]), "action.bimanual": actions})
