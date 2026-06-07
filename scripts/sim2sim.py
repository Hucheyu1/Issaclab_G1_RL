"""Unified sim-to-sim runner for motion-tracking policies in MuJoCo.

The G1 path is set up for the exported policies in this repository:

    python scripts/sim2sim.py --robot g1 --headless --max_steps 50
    python scripts/sim2sim.py --robot g1 --motion_file datasets/extend_datasets/lafan1_dataset/g1/train/

Both older single-input policies (``obs``) and GAEMimic robot policies with
split ONNX inputs (``robot_command``, ``motion_anchor_ori_b``, ...) are
supported.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime

SIMULATION_DURATION = 300.0
SIMULATION_DT = 0.002
CONTROL_DECIMATION = 10

DEFAULT_G1_POLICY_PATH = Path(
    "/root/gpufree-data/lab_lecture/wbc_robot/logs/rsl_rl/multi_g1_flat/2026-05-20_14-58-09/exported/policy.onnx"
)
DEFAULT_G1_MOTION_PATH = Path("datasets/extend_datasets/lafan1_dataset/g1/train/")
DEFAULT_G1_MOTION_NAME = "dance1_subject1.npz"
DEFAULT_G1_XML_CANDIDATES = (
    Path("unitree_model/G1/29dof/scene_29dof.xml"),
    Path("unitree_model/G1/29dof/g1_29dof.xml"),
)


G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _first_existing_path(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        expanded = path.expanduser()
        if expanded.exists():
            return expanded
    return paths[0]


def _build_g1_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    armature_5020 = 0.003609725
    armature_7520_14 = 0.010177520
    armature_7520_22 = 0.025101925
    armature_4010 = 0.00425
    natural_freq = 10.0 * 2.0 * np.pi
    damping_ratio = 2.0

    stiffness_5020 = armature_5020 * natural_freq**2
    stiffness_7520_14 = armature_7520_14 * natural_freq**2
    stiffness_7520_22 = armature_7520_22 * natural_freq**2
    stiffness_4010 = armature_4010 * natural_freq**2

    damping_5020 = 2.0 * damping_ratio * armature_5020 * natural_freq
    damping_7520_14 = 2.0 * damping_ratio * armature_7520_14 * natural_freq
    damping_7520_22 = 2.0 * damping_ratio * armature_7520_22 * natural_freq
    damping_4010 = 2.0 * damping_ratio * armature_4010 * natural_freq

    default_pos = {name: 0.0 for name in G1_JOINT_NAMES}
    stiffness = {}
    damping = {}
    effort = {}

    for side in ("left", "right"):
        default_pos[f"{side}_hip_pitch_joint"] = -0.312
        default_pos[f"{side}_knee_joint"] = 0.669
        default_pos[f"{side}_ankle_pitch_joint"] = -0.363
        default_pos[f"{side}_shoulder_pitch_joint"] = 0.2
        default_pos[f"{side}_elbow_joint"] = 0.6

        for suffix in ("hip_pitch", "hip_yaw"):
            name = f"{side}_{suffix}_joint"
            stiffness[name] = stiffness_7520_14
            damping[name] = damping_7520_14
            effort[name] = 88.0
        for suffix in ("hip_roll", "knee"):
            name = f"{side}_{suffix}_joint"
            stiffness[name] = stiffness_7520_22
            damping[name] = damping_7520_22
            effort[name] = 139.0
        for suffix in ("ankle_pitch", "ankle_roll"):
            name = f"{side}_{suffix}_joint"
            stiffness[name] = 2.0 * stiffness_5020
            damping[name] = 2.0 * damping_5020
            effort[name] = 50.0
        for suffix in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"):
            name = f"{side}_{suffix}_joint"
            stiffness[name] = stiffness_5020
            damping[name] = damping_5020
            effort[name] = 25.0
        for suffix in ("wrist_pitch", "wrist_yaw"):
            name = f"{side}_{suffix}_joint"
            stiffness[name] = stiffness_4010
            damping[name] = damping_4010
            effort[name] = 5.0

    default_pos["left_shoulder_roll_joint"] = 0.2
    default_pos["right_shoulder_roll_joint"] = -0.2

    stiffness["waist_yaw_joint"] = stiffness_7520_14
    damping["waist_yaw_joint"] = damping_7520_14
    effort["waist_yaw_joint"] = 88.0
    for name in ("waist_roll_joint", "waist_pitch_joint"):
        stiffness[name] = 2.0 * stiffness_5020
        damping[name] = 2.0 * damping_5020
        effort[name] = 50.0

    default_array = np.array([default_pos[name] for name in G1_JOINT_NAMES], dtype=np.float32)
    stiffness_array = np.array([stiffness[name] for name in G1_JOINT_NAMES], dtype=np.float32)
    damping_array = np.array([damping[name] for name in G1_JOINT_NAMES], dtype=np.float32)
    action_scale = np.array([0.25 * effort[name] / stiffness[name] for name in G1_JOINT_NAMES], dtype=np.float32)
    return default_array, stiffness_array, damping_array, action_scale


G1_DEFAULT_JOINT_POS, G1_STIFFNESS, G1_DAMPING, G1_ACTION_SCALE = _build_g1_arrays()


ROBOT_CONFIGS = {
    "hi": {
        "num_actions": 23,
        "num_obs": 124,
        "reference_body": "base_link",
        "default_xml": None,
        "joint_names": [
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_hip_thigh_joint",
            "l_hip_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_hip_thigh_joint",
            "r_hip_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "waist_yaw_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ],
        "motion_body_index": 0,
    },
    "pi_plus": {
        "num_actions": 22,
        "num_obs": 119,
        "reference_body": "base_link",
        "default_xml": None,
        "joint_names": [
            "l_hip_pitch_joint",
            "l_hip_roll_joint",
            "l_thigh_joint",
            "l_calf_joint",
            "l_ankle_pitch_joint",
            "l_ankle_roll_joint",
            "l_shoulder_pitch_joint",
            "l_shoulder_roll_joint",
            "l_upper_arm_joint",
            "l_elbow_joint",
            "l_wrist_joint",
            "r_hip_pitch_joint",
            "r_hip_roll_joint",
            "r_thigh_joint",
            "r_calf_joint",
            "r_ankle_pitch_joint",
            "r_ankle_roll_joint",
            "r_shoulder_pitch_joint",
            "r_shoulder_roll_joint",
            "r_upper_arm_joint",
            "r_elbow_joint",
            "r_wrist_joint",
        ],
        "motion_body_index": 0,
    },
    "g1": {
        "num_actions": 29,
        "num_obs": 154,
        "reference_body": "pelvis",
        "default_xml": str(_first_existing_path(DEFAULT_G1_XML_CANDIDATES)),
        "joint_names": G1_JOINT_NAMES,
        "default_joint_pos": G1_DEFAULT_JOINT_POS,
        "joint_stiffness": G1_STIFFNESS,
        "joint_damping": G1_DAMPING,
        "action_scale": G1_ACTION_SCALE,
        "base_height": 0.76,
        "motion_body_index": 0,
    },
}


def matrix_from_quat_np(quat: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quat, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    two_s = 2.0 / np.dot(q, q)
    return np.array(
        [
            [1 - two_s * (y * y + z * z), two_s * (x * y - z * w), two_s * (x * z + y * w)],
            [two_s * (x * y + z * w), 1 - two_s * (x * x + z * z), two_s * (y * z - x * w)],
            [two_s * (x * z - y * w), two_s * (y * z + x * w), 1 - two_s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_conjugate_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.asarray(q1, dtype=np.float64)
    w2, x2, y2, z2 = np.asarray(q2, dtype=np.float64)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_inv_np(q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return quat_conjugate_np(q) / max(float(np.dot(q, q)), eps)


def quat_rotate_inverse_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by inverse quaternion q, both using MuJoCo/Isaac wxyz convention."""
    rot = matrix_from_quat_np(q)
    return rot.T @ np.asarray(v, dtype=np.float64)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def _parse_float_csv(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    return np.array([float(x.strip()) for x in value.split(",") if x.strip()], dtype=np.float32)


def _parse_str_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [x.strip() for x in value.split(",") if x.strip()]


def _resolve_motion_file(path: str | Path, preferred_name: str = DEFAULT_G1_MOTION_NAME) -> Path:
    motion_path = Path(path).expanduser()
    if motion_path.is_dir():
        preferred = motion_path / preferred_name
        if preferred.exists():
            print(f"[INFO]: Motion directory provided; using {preferred}")
            return preferred
        npz_files = sorted(motion_path.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(f"No .npz motion files found in directory: {motion_path}")
        print(f"[INFO]: Motion directory provided; using first file {npz_files[0]}")
        return npz_files[0]
    if motion_path.suffix != ".npz":
        with_suffix = motion_path.with_suffix(".npz")
        if with_suffix.exists():
            return with_suffix
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
    return motion_path


def _resolve_policy_path(path: str | Path) -> Path:
    policy_path = Path(path).expanduser()
    if policy_path.is_dir():
        policy_path = policy_path / "policy.onnx"
    if not policy_path.exists():
        raise FileNotFoundError(f"ONNX policy does not exist: {policy_path}")
    return policy_path


def _load_mujoco_model(xml_path: str | Path) -> mujoco.MjModel:
    xml = Path(xml_path).expanduser()
    if not xml.exists():
        raise FileNotFoundError(f"MuJoCo XML does not exist: {xml}")
    cwd = os.getcwd()
    try:
        os.chdir(xml.parent)
        model = mujoco.MjModel.from_xml_path(xml.name)
    except Exception as exc:
        msg = (
            f"Failed to load MuJoCo XML: {xml}\n"
            "If this is the repository copy under unitree_model/G1/29dof, the STL meshes may be missing. "
            "Use /root/unitree_mujoco/unitree_robots/g1/scene_29dof.xml or install the Unitree mesh folder."
        )
        raise RuntimeError(msg) from exc
    finally:
        os.chdir(cwd)
    return model


def _metadata_dict(model: onnx.ModelProto) -> dict[str, str]:
    return {prop.key: prop.value for prop in model.metadata_props}


def _policy_metadata_or_config(
    model: onnx.ModelProto, config: dict
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    metadata = _metadata_dict(model)
    joint_names = _parse_str_csv(metadata.get("joint_names")) or list(config.get("joint_names", []))
    default_pos = _parse_float_csv(metadata.get("default_joint_pos"))
    stiffness = _parse_float_csv(metadata.get("joint_stiffness"))
    damping = _parse_float_csv(metadata.get("joint_damping"))
    action_scale = _parse_float_csv(metadata.get("action_scale"))

    default_pos = default_pos if default_pos is not None else config.get("default_joint_pos")
    stiffness = stiffness if stiffness is not None else config.get("joint_stiffness")
    damping = damping if damping is not None else config.get("joint_damping")
    action_scale = action_scale if action_scale is not None else config.get("action_scale")

    missing = [
        name
        for name, value in (
            ("joint_names", joint_names),
            ("default_joint_pos", default_pos),
            ("joint_stiffness", stiffness),
            ("joint_damping", damping),
            ("action_scale", action_scale),
        )
        if value is None or len(value) == 0
    ]
    if missing:
        raise RuntimeError("Policy metadata is missing and no local fallback is available for: " + ", ".join(missing))

    num_actions = config["num_actions"]
    arrays = [np.asarray(default_pos), np.asarray(stiffness), np.asarray(damping), np.asarray(action_scale)]
    if len(joint_names) != num_actions or any(len(arr) != num_actions for arr in arrays):
        raise ValueError(
            f"Expected {num_actions} policy joints/arrays, got "
            f"joint_names={len(joint_names)}, default={len(arrays[0])}, stiffness={len(arrays[1])}, "
            f"damping={len(arrays[2])}, action_scale={len(arrays[3])}"
        )

    return (
        joint_names,
        arrays[0].astype(np.float32),
        arrays[1].astype(np.float32),
        arrays[2].astype(np.float32),
        arrays[3].astype(np.float32),
    )


def _actuator_joint_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id][0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is not None:
            names.append(name)
    return names


def _reorder(values: np.ndarray, source_names: list[str], target_names: list[str], label: str) -> np.ndarray:
    index = {name: i for i, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in index]
    if missing:
        raise ValueError(f"Cannot remap {label}; missing names: {missing}")
    return np.asarray([values[index[name]] for name in target_names], dtype=np.float32)


def _joint_positions(data: mujoco.MjData, model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        values.append(data.qpos[model.jnt_qposadr[joint_id]])
    return np.asarray(values, dtype=np.float32)


def _joint_velocities(data: mujoco.MjData, model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    values = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        values.append(data.qvel[model.jnt_dofadr[joint_id]])
    return np.asarray(values, dtype=np.float32)


def _write_joint_positions(
    data: mujoco.MjData, model: mujoco.MjModel, joint_names: list[str], values: np.ndarray
) -> None:
    for name, value in zip(joint_names, values):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint_id]] = value


def _write_joint_velocities(
    data: mujoco.MjData, model: mujoco.MjModel, joint_names: list[str], values: np.ndarray
) -> None:
    for name, value in zip(joint_names, values):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qvel[model.jnt_dofadr[joint_id]] = value


def _sensor_data(model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...]) -> np.ndarray | None:
    for name in names:
        sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id != -1:
            return np.asarray(data.sensor(name).data, dtype=np.float64).copy()
    return None


def _robot_orientation_and_gyro(
    model: mujoco.MjModel, data: mujoco.MjData, reference_body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    quat = _sensor_data(model, data, ("imu_quat", "orientation", "secondary_imu_quat"))
    if quat is None or quat.shape[0] != 4:
        quat = np.asarray(data.xquat[reference_body_id], dtype=np.float64).copy()
    quat = quat / np.linalg.norm(quat)

    omega = _sensor_data(model, data, ("imu_gyro", "angular-velocity", "secondary_imu_gyro"))
    if omega is None or omega.shape[0] != 3:
        omega = quat_rotate_inverse_np(quat, data.qvel[3:6])
    return quat.astype(np.float32), omega.astype(np.float32)


def _frame_idx(timestep: int, num_frames: int, loop: bool) -> int:
    if num_frames <= 0:
        raise ValueError("Motion file has no frames.")
    if loop:
        return timestep % num_frames
    return min(timestep, num_frames - 1)


def _future_indices(timestep: int, num_frames: int, loop: bool, frames: int = 10, interval: int = 2) -> np.ndarray:
    return np.array([_frame_idx(timestep + interval * i, num_frames, loop) for i in range(frames)], dtype=np.int64)


def _motion_reference_ori_b(robot_quat_w: np.ndarray, motion_quat_w: np.ndarray) -> np.ndarray:
    rel_quat = quat_mul_np(quat_inv_np(robot_quat_w), motion_quat_w)
    rel_quat = rel_quat / np.linalg.norm(rel_quat)
    mat = matrix_from_quat_np(rel_quat)
    return mat[:, :2].reshape(-1).astype(np.float32)


def _command_obs(motion: np.lib.npyio.NpzFile, idx: int) -> np.ndarray:
    return np.concatenate((motion["joint_pos"][idx], motion["joint_vel"][idx]), axis=0).astype(np.float32)


def _robot_command(motion: np.lib.npyio.NpzFile, indices: np.ndarray) -> np.ndarray:
    return motion["joint_pos"][indices].reshape(-1).astype(np.float32)


def _human_command(motion: np.lib.npyio.NpzFile, indices: np.ndarray) -> np.ndarray:
    if "smplx_pose_body" not in motion:
        raise KeyError("ONNX expects human_command, but the motion file has no smplx_pose_body key.")
    return motion["smplx_pose_body"][indices].reshape(-1).astype(np.float32)


def _keypoints_command(motion: np.lib.npyio.NpzFile, indices: np.ndarray) -> np.ndarray:
    if "robot_keypoints_trans" not in motion or "robot_keypoints_rot" not in motion:
        raise KeyError("ONNX expects keypoints_command, but the motion file has no robot_keypoints_* keys.")
    keypoints = np.concatenate(
        (motion["robot_keypoints_trans"][indices], motion["robot_keypoints_rot"][indices]), axis=-1
    )
    return keypoints.reshape(-1).astype(np.float32)


def _as_onnx_input(name: str, value: np.ndarray, expected_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(1, -1)
    if expected_dim is not None and expected_dim > 0 and arr.shape[1] != expected_dim:
        raise ValueError(f"ONNX input {name!r} expected dim {expected_dim}, got {arr.shape[1]}")
    return arr


def _onnx_input_dim(input_info) -> int | None:
    shape = input_info.shape
    if len(shape) < 2:
        return None
    dim = shape[1]
    return int(dim) if isinstance(dim, int) else None


def _build_policy_feed(
    input_infos,
    motion: np.lib.npyio.NpzFile,
    timestep: int,
    loop: bool,
    num_frames: int,
    robot_quat_w: np.ndarray,
    motion_quat_w: np.ndarray,
    omega: np.ndarray,
    joint_pos_train: np.ndarray,
    joint_vel_train: np.ndarray,
    default_joint_pos_train: np.ndarray,
    action_buffer_train: np.ndarray,
) -> dict[str, np.ndarray]:
    idx = _frame_idx(timestep, num_frames, loop)
    future = _future_indices(timestep, num_frames, loop)
    motion_ref_ori_b = _motion_reference_ori_b(robot_quat_w, motion_quat_w)
    joint_pos_rel = (joint_pos_train - default_joint_pos_train).astype(np.float32)

    pieces = {
        "command": _command_obs(motion, idx),
        "robot_command": _robot_command(motion, future),
        "human_command": lambda: _human_command(motion, future),
        "keypoints_command": lambda: _keypoints_command(motion, future),
        "motion_anchor_ori_b": motion_ref_ori_b,
        "base_ang_vel": omega.astype(np.float32),
        "joint_pos": joint_pos_rel,
        "joint_vel": joint_vel_train.astype(np.float32),
        "actions": action_buffer_train.astype(np.float32),
        "time_step": np.array([[idx]], dtype=np.float32),
    }

    feed = {}
    for input_info in input_infos:
        name = input_info.name
        dim = _onnx_input_dim(input_info)
        base_name = name.split(".")[0]

        if name == "obs":
            obs = np.concatenate(
                (
                    pieces["command"],
                    pieces["motion_anchor_ori_b"],
                    pieces["base_ang_vel"],
                    pieces["joint_pos"],
                    pieces["joint_vel"],
                    pieces["actions"],
                ),
                axis=0,
            )
            feed[name] = _as_onnx_input(name, obs, dim)
        elif name == "time_step":
            feed[name] = pieces["time_step"]
        elif base_name in pieces:
            value = pieces[base_name]
            if callable(value):
                value = value()
            feed[name] = _as_onnx_input(name, value, dim)
        else:
            raise ValueError(f"Unsupported ONNX input {name!r}. Inputs: {[i.name for i in input_infos]}")

    return feed


def _initialize_from_motion(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    motion: np.lib.npyio.NpzFile,
    motion_body_idx: int,
    policy_joint_names: list[str],
    actuator_joint_names: list[str],
    default_joint_pos_train: np.ndarray,
    init_from_motion: bool,
    base_height: float,
) -> None:
    if init_from_motion:
        data.qpos[:3] = motion["body_pos_w"][0, motion_body_idx]
        data.qpos[3:7] = motion["body_quat_w"][0, motion_body_idx]
        joint_pos_actuator = _reorder(
            motion["joint_pos"][0], policy_joint_names, actuator_joint_names, "initial joint pos"
        )
        _write_joint_positions(data, model, actuator_joint_names, joint_pos_actuator)
        if "body_lin_vel_w" in motion:
            data.qvel[:3] = motion["body_lin_vel_w"][0, motion_body_idx]
        if "body_ang_vel_w" in motion:
            data.qvel[3:6] = motion["body_ang_vel_w"][0, motion_body_idx]
        joint_vel_actuator = _reorder(
            motion["joint_vel"][0], policy_joint_names, actuator_joint_names, "initial joint vel"
        )
        _write_joint_velocities(data, model, actuator_joint_names, joint_vel_actuator)
    else:
        data.qpos[2] = base_height
        data.qpos[3] = 1.0
        default_actuator = _reorder(
            default_joint_pos_train, policy_joint_names, actuator_joint_names, "default joint pos"
        )
        _write_joint_positions(data, model, actuator_joint_names, default_actuator)
    mujoco.mj_forward(model, data)


def _save_motion_json(motion: np.lib.npyio.NpzFile, motion_file: Path) -> None:
    motion_dict = {
        "body_pos_w": motion["body_pos_w"].tolist(),
        "body_quat_w": motion["body_quat_w"].tolist(),
        "joint_pos": motion["joint_pos"].tolist(),
        "joint_vel": motion["joint_vel"].tolist(),
    }
    json_filename = motion_file.with_suffix(".json")
    with json_filename.open("w") as f:
        json.dump(motion_dict, f, indent=2)
    print(f"[INFO]: Motion data saved to: {json_filename}")


def run_simulation(
    robot_type: str,
    motion_file: str | Path,
    xml_path: str | Path,
    policy_path: str | Path,
    save_json: bool = False,
    loop: bool = False,
    headless: bool = False,
    duration: float = SIMULATION_DURATION,
    dt: float = SIMULATION_DT,
    decimation: int = CONTROL_DECIMATION,
    max_steps: int | None = None,
    init_from_motion: bool = True,
) -> None:
    config = ROBOT_CONFIGS[robot_type]
    motion_file = _resolve_motion_file(motion_file)
    policy_path = _resolve_policy_path(policy_path)

    print(f"[INFO]: Robot: {robot_type}")
    print(f"[INFO]: Motion file: {motion_file}")
    print(f"[INFO]: XML path: {xml_path}")
    print(f"[INFO]: Policy path: {policy_path}")

    motion = np.load(motion_file)
    required_motion_keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
    missing_motion_keys = [key for key in required_motion_keys if key not in motion]
    if missing_motion_keys:
        raise KeyError(f"Motion file is missing keys: {missing_motion_keys}")

    num_frames = min(
        motion["joint_pos"].shape[0],
        motion["joint_vel"].shape[0],
        motion["body_pos_w"].shape[0],
        motion["body_quat_w"].shape[0],
    )
    if save_json:
        _save_motion_json(motion, motion_file)

    onnx_model = onnx.load(policy_path)
    policy_joint_names, default_joint_pos, stiffness, damping, action_scale = _policy_metadata_or_config(
        onnx_model, config
    )

    model = _load_mujoco_model(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = dt

    actuator_joint_names = _actuator_joint_names(model)
    if len(actuator_joint_names) != config["num_actions"]:
        raise ValueError(
            f"Expected {config['num_actions']} actuated joints, got {len(actuator_joint_names)}: {actuator_joint_names}"
        )

    default_joint_pos_actuator = _reorder(
        default_joint_pos, policy_joint_names, actuator_joint_names, "default joint pos"
    )
    stiffness_actuator = _reorder(stiffness, policy_joint_names, actuator_joint_names, "joint stiffness")
    damping_actuator = _reorder(damping, policy_joint_names, actuator_joint_names, "joint damping")

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config["reference_body"])
    if body_id == -1:
        raise ValueError(f"Body {config['reference_body']} not found in MuJoCo model.")

    session = onnxruntime.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_infos = session.get_inputs()
    output_name = session.get_outputs()[0].name
    print(f"[INFO]: ONNX inputs: {[(i.name, i.shape) for i in input_infos]}")
    print(f"[INFO]: ONNX output: {output_name}")
    print(f"[INFO]: Policy joint order: {policy_joint_names}")
    print(f"[INFO]: MuJoCo actuator joint order: {actuator_joint_names}")

    _initialize_from_motion(
        data,
        model,
        motion,
        config["motion_body_index"],
        policy_joint_names,
        actuator_joint_names,
        default_joint_pos,
        init_from_motion,
        config.get("base_height", 0.75),
    )

    num_actions = config["num_actions"]
    action_buffer = np.zeros((num_actions,), dtype=np.float32)
    target_dof_pos_actuator = default_joint_pos_actuator.copy()
    timestep = 0
    counter = 0
    total_steps = max_steps if max_steps is not None else int(duration / dt)

    def step_once() -> None:
        nonlocal action_buffer, target_dof_pos_actuator, timestep, counter

        qpos_actuator = _joint_positions(data, model, actuator_joint_names)
        qvel_actuator = _joint_velocities(data, model, actuator_joint_names)
        tau = pd_control(
            target_dof_pos_actuator,
            qpos_actuator,
            stiffness_actuator,
            np.zeros_like(damping_actuator),
            qvel_actuator,
            damping_actuator,
        )
        if model.actuator_ctrllimited.any():
            tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        counter += 1

        if counter % decimation != 0:
            return

        idx = _frame_idx(timestep, num_frames, loop)
        robot_quat_w, omega = _robot_orientation_and_gyro(model, data, body_id)
        motion_quat_w = motion["body_quat_w"][idx, config["motion_body_index"]]
        joint_pos_train = _reorder(
            _joint_positions(data, model, actuator_joint_names),
            actuator_joint_names,
            policy_joint_names,
            "current joint pos",
        )
        joint_vel_train = _reorder(
            _joint_velocities(data, model, actuator_joint_names),
            actuator_joint_names,
            policy_joint_names,
            "current joint vel",
        )

        feed = _build_policy_feed(
            input_infos,
            motion,
            timestep,
            loop,
            num_frames,
            robot_quat_w,
            motion_quat_w,
            omega,
            joint_pos_train,
            joint_vel_train,
            default_joint_pos,
            action_buffer,
        )

        action = np.asarray(session.run([output_name], feed)[0], dtype=np.float32).reshape(-1)
        if action.shape[0] != num_actions:
            raise ValueError(f"Policy returned {action.shape[0]} actions, expected {num_actions}")

        action_buffer = action.copy()
        target_train = action * action_scale + default_joint_pos
        target_dof_pos_actuator = _reorder(target_train, policy_joint_names, actuator_joint_names, "target joint pos")

        if loop or timestep + 1 < num_frames:
            timestep += 1

    if headless:
        for _ in range(total_steps):
            step_once()
        print(f"[INFO]: Headless simulation completed: steps={total_steps}, policy_steps={counter // decimation}")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < duration:
            step_start = time.time()
            step_once()
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a motion policy in MuJoCo for sim-to-sim validation.")
    parser.add_argument("--robot", type=str, choices=["hi", "pi_plus", "g1"], default="g1", help="Robot type.")
    parser.add_argument(
        "--motion_file",
        type=str,
        default=str(DEFAULT_G1_MOTION_PATH),
        help="Path to a motion NPZ file, or a directory containing NPZ files.",
    )
    parser.add_argument(
        "--motion_name",
        type=str,
        default=DEFAULT_G1_MOTION_NAME,
        help="Preferred NPZ filename when --motion_file is a directory.",
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        default=None,
        help="Path to the MuJoCo XML file. Defaults to the G1 29dof Unitree MuJoCo scene when available.",
    )
    parser.add_argument(
        "--policy_path",
        type=str,
        default=str(DEFAULT_G1_POLICY_PATH),
        help="Path to ONNX policy, or a directory containing policy.onnx.",
    )
    parser.add_argument("--save_json", action="store_true", help="Save motion data next to the NPZ as JSON.")
    parser.add_argument("--loop", action="store_true", help="Loop motion/policy when reaching the end of sequence.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a MuJoCo viewer.")
    parser.add_argument("--duration", type=float, default=SIMULATION_DURATION, help="Simulation duration in seconds.")
    parser.add_argument("--dt", type=float, default=SIMULATION_DT, help="MuJoCo physics timestep.")
    parser.add_argument("--decimation", type=int, default=CONTROL_DECIMATION, help="Policy control decimation.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum physics steps for headless smoke tests.")
    parser.add_argument(
        "--no_init_from_motion",
        action="store_true",
        help="Start from the nominal standing pose instead of the first motion frame.",
    )

    args = parser.parse_args()
    config = ROBOT_CONFIGS[args.robot]
    xml_path = args.xml_path or config.get("default_xml")
    if not xml_path:
        raise ValueError(f"--xml_path is required for robot {args.robot}")

    motion_file = args.motion_file
    if Path(motion_file).expanduser().is_dir():
        motion_file = _resolve_motion_file(motion_file, args.motion_name)

    run_simulation(
        args.robot,
        motion_file,
        xml_path,
        args.policy_path,
        save_json=args.save_json,
        loop=args.loop,
        headless=args.headless,
        duration=args.duration,
        dt=args.dt,
        decimation=args.decimation,
        max_steps=args.max_steps,
        init_from_motion=not args.no_init_from_motion,
    )


if __name__ == "__main__":
    main()
