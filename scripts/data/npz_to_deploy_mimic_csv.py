"""Convert extended G1 motion NPZ files into deploy mimic CSV files.

Examples:
    python scripts/data/npz_to_deploy_mimic_csv.py \
        --motion datasets/extend_datasets/lafan1_dataset/g1/train/run1_subject2.npz

    python scripts/data/npz_to_deploy_mimic_csv.py \
        --motion datasets/extend_datasets/lafan1_dataset/g1/train/walk1_subject1.npz \
        --policy multi_g1_flat gaemimic_robot

The generated files are centralized under:
    deploy/robots/g1_29dof/config/policy/mimic/data/base
    deploy/robots/g1_29dof/config/policy/mimic/data/human
    deploy/robots/g1_29dof/config/policy/mimic/data/keypoints
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


JOINT_IDS_MAP = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)

POLICY_KINDS = {
    "multi_g1_flat": "base",
    "gaemimic_policy": "base",
    "gaemimic_robot": "base",
    "gaemimic_human": "human",
    "gaemimic_keypoints": "keypoints",
}


def _fps_value(data: np.lib.npyio.NpzFile) -> float:
    if "fps" not in data:
        raise KeyError("Motion NPZ is missing key: fps")
    fps = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
    if fps.size == 0:
        raise ValueError("Motion NPZ has an empty fps field.")
    return float(fps[0])


def _fps_label(fps: float) -> str:
    if np.isclose(fps, round(fps)):
        return f"{int(round(fps))}hz"
    return f"{fps:g}hz"


def _base_columns(data: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("body_pos_w", "body_quat_w", "joint_pos"):
        if key not in data:
            raise KeyError(f"Motion NPZ is missing key: {key}")

    root_pos = data["body_pos_w"][:, 0, :]
    quat_wxyz = data["body_quat_w"][:, 0, :]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    # The NPZ stores joints in IsaacLab/policy order. Deploy MotionLoader stores
    # CSV joint columns in Unitree motor order, then maps them back by joint_ids_map.
    joint_motor = np.empty_like(data["joint_pos"])
    joint_motor[:, JOINT_IDS_MAP] = data["joint_pos"]
    return np.concatenate([root_pos, quat_xyzw, joint_motor], axis=1)


def _extra_columns(data: np.lib.npyio.NpzFile, kind: str) -> np.ndarray | None:
    if kind == "base":
        return None
    if kind == "human":
        if "smplx_pose_body" not in data:
            raise KeyError("human policy CSV requires smplx_pose_body in the NPZ.")
        return data["smplx_pose_body"].reshape(data["joint_pos"].shape[0], -1)
    if kind == "keypoints":
        for key in ("robot_keypoints_trans", "robot_keypoints_rot"):
            if key not in data:
                raise KeyError(f"keypoints policy CSV requires {key} in the NPZ.")
        keypoints = np.concatenate([data["robot_keypoints_trans"], data["robot_keypoints_rot"]], axis=-1)
        return keypoints.reshape(data["joint_pos"].shape[0], -1)
    raise ValueError(f"Unknown policy kind: {kind}")


def _deploy_columns(data: np.lib.npyio.NpzFile, kind: str) -> np.ndarray:
    base = _base_columns(data)
    extra = _extra_columns(data, kind)
    if extra is None:
        return base
    return np.concatenate([base, extra], axis=1)


def convert_motion(motion_path: Path, deploy_mimic_root: Path, policies: list[str]) -> None:
    data = np.load(motion_path)
    fps = _fps_value(data)
    filename = f"{motion_path.stem}_{_fps_label(fps)}.csv"

    kinds = []
    for policy in policies:
        kind = POLICY_KINDS[policy]
        if kind not in kinds:
            kinds.append(kind)

    for kind in kinds:
        columns = _deploy_columns(data, kind)
        out_path = deploy_mimic_root / "data" / kind / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_path, columns, delimiter=",", fmt="%.6f")
        print(f"wrote {out_path} shape={columns.shape}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert extended G1 motion NPZ files to deploy mimic CSV files.")
    parser.add_argument("--motion", nargs="+", required=True, help="Input extended G1 motion NPZ file(s).")
    parser.add_argument(
        "--deploy_mimic_root",
        default="deploy/robots/g1_29dof/config/policy/mimic",
        help="Root directory containing deploy mimic policy folders.",
    )
    parser.add_argument(
        "--policy",
        nargs="+",
        choices=sorted(POLICY_KINDS),
        default=sorted(POLICY_KINDS),
        help="Deploy policy folders to generate CSVs for.",
    )
    args = parser.parse_args()

    root = Path(args.deploy_mimic_root).expanduser()
    for motion in args.motion:
        convert_motion(Path(motion).expanduser(), root, args.policy)


if __name__ == "__main__":
    main()
