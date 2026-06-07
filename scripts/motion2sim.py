"""Replay a motion NPZ directly on the MuJoCo G1 model.

Examples:
    # Exact kinematic replay, one motion frame per rendered frame.
    python scripts/motion2sim.py --motion_file datasets/extend_datasets/lafan1_dataset/g1/train/dance1_subject1.npz

    # Smoke test without opening the viewer.
    python scripts/motion2sim.py --headless --max_frames 20

    # Dynamic replay: initialize from the motion, then let MuJoCo physics run.
    python scripts/motion2sim.py --mode pd --headless --max_steps 200

Kinematic replay is useful for checking whether an NPZ is readable and whether
joint/root coordinates are wired correctly. It is not a physical simulation:
the floating base and joints are overwritten from the file.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from sim2sim import (
    CONTROL_DECIMATION,
    DEFAULT_G1_MOTION_NAME,
    DEFAULT_G1_MOTION_PATH,
    DEFAULT_G1_XML_CANDIDATES,
    G1_DAMPING,
    G1_JOINT_NAMES,
    G1_STIFFNESS,
    _actuator_joint_names,
    _first_existing_path,
    _joint_positions,
    _joint_velocities,
    _load_mujoco_model,
    _reorder,
    _resolve_motion_file,
    _write_joint_positions,
    _write_joint_velocities,
    pd_control,
)

DEFAULT_PHYSICS_DT = 0.002
DEFAULT_REFERENCE_BODY_INDEX = 0


def _motion_fps(motion: np.lib.npyio.NpzFile) -> float:
    if "fps" not in motion:
        return 50.0
    fps = np.asarray(motion["fps"]).reshape(-1)
    if fps.size == 0:
        return 50.0
    return float(fps[0])


def _num_motion_frames(motion: np.lib.npyio.NpzFile) -> int:
    required_keys = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
    missing = [key for key in required_keys if key not in motion]
    if missing:
        raise KeyError(f"Motion file is missing keys: {missing}")
    return min(
        motion["joint_pos"].shape[0],
        motion["joint_vel"].shape[0],
        motion["body_pos_w"].shape[0],
        motion["body_quat_w"].shape[0],
    )


def _frame_idx(frame: int, num_frames: int, loop: bool) -> int:
    if num_frames <= 0:
        raise ValueError("Motion file has no frames.")
    if loop:
        return frame % num_frames
    return min(frame, num_frames - 1)


def _write_root_state(
    data: mujoco.MjData,
    motion: np.lib.npyio.NpzFile,
    frame_idx: int,
    body_idx: int,
    root_height_offset: float = 0.0,
) -> None:
    data.qpos[:3] = motion["body_pos_w"][frame_idx, body_idx]
    data.qpos[2] += root_height_offset
    data.qpos[3:7] = motion["body_quat_w"][frame_idx, body_idx]
    if "body_lin_vel_w" in motion:
        data.qvel[:3] = motion["body_lin_vel_w"][frame_idx, body_idx]
    if "body_ang_vel_w" in motion:
        data.qvel[3:6] = motion["body_ang_vel_w"][frame_idx, body_idx]


def write_motion_frame_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: np.lib.npyio.NpzFile,
    frame_idx: int,
    actuator_joint_names: list[str],
    *,
    body_idx: int = DEFAULT_REFERENCE_BODY_INDEX,
    write_root: bool = True,
    root_height_offset: float = 0.0,
) -> None:
    """Write one motion frame into MuJoCo qpos/qvel and run mj_forward."""
    if write_root:
        _write_root_state(data, motion, frame_idx, body_idx, root_height_offset)

    joint_pos = _reorder(motion["joint_pos"][frame_idx], G1_JOINT_NAMES, actuator_joint_names, "motion joint pos")
    joint_vel = _reorder(motion["joint_vel"][frame_idx], G1_JOINT_NAMES, actuator_joint_names, "motion joint vel")
    _write_joint_positions(data, model, actuator_joint_names, joint_pos)
    _write_joint_velocities(data, model, actuator_joint_names, joint_vel)
    mujoco.mj_forward(model, data)


def _kinematic_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: np.lib.npyio.NpzFile,
    frame_counter: int,
    num_frames: int,
    actuator_joint_names: list[str],
    loop: bool,
    write_root: bool,
    root_height_offset: float,
) -> int:
    idx = _frame_idx(frame_counter, num_frames, loop)
    write_motion_frame_to_mujoco(
        model,
        data,
        motion,
        idx,
        actuator_joint_names,
        write_root=write_root,
        root_height_offset=root_height_offset,
    )
    return idx


def _pd_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: np.lib.npyio.NpzFile,
    step_counter: int,
    num_frames: int,
    steps_per_motion_frame: int,
    actuator_joint_names: list[str],
    stiffness_actuator: np.ndarray,
    damping_actuator: np.ndarray,
    loop: bool,
    pin_root: bool,
    root_height_offset: float,
) -> int:
    idx = _frame_idx(step_counter // steps_per_motion_frame, num_frames, loop)
    if pin_root:
        _write_root_state(data, motion, idx, DEFAULT_REFERENCE_BODY_INDEX, root_height_offset)

    target_q = _reorder(motion["joint_pos"][idx], G1_JOINT_NAMES, actuator_joint_names, "target joint pos")
    current_q = _joint_positions(data, model, actuator_joint_names)
    current_dq = _joint_velocities(data, model, actuator_joint_names)
    tau = pd_control(
        target_q,
        current_q,
        stiffness_actuator,
        np.zeros_like(damping_actuator),
        current_dq,
        damping_actuator,
    )
    if model.actuator_ctrllimited.any():
        tau = np.clip(tau, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
    data.ctrl[:] = tau
    mujoco.mj_step(model, data)
    return idx


def run_motion2sim(
    motion_file: str | Path,
    xml_path: str | Path,
    *,
    mode: str = "kinematic",
    loop: bool = False,
    headless: bool = False,
    max_frames: int | None = None,
    max_steps: int | None = None,
    physics_dt: float = DEFAULT_PHYSICS_DT,
    speed: float = 1.0,
    write_root: bool = True,
    init_root: bool = True,
    pin_root: bool = False,
    root_height_offset: float = 0.0,
    kp_scale: float = 1.0,
    kd_scale: float = 1.0,
    render_decimation: int | None = None,
) -> None:
    motion_path = _resolve_motion_file(motion_file)
    motion = np.load(motion_path)
    num_frames = _num_motion_frames(motion)
    fps = _motion_fps(motion)
    motion_dt = 1.0 / fps

    model = _load_mujoco_model(xml_path)
    data = mujoco.MjData(model)
    model.opt.timestep = physics_dt

    actuator_joint_names = _actuator_joint_names(model)
    if len(actuator_joint_names) != len(G1_JOINT_NAMES):
        raise ValueError(f"Expected 29 G1 actuated joints, got {len(actuator_joint_names)}: {actuator_joint_names}")

    stiffness_actuator = kp_scale * _reorder(G1_STIFFNESS, G1_JOINT_NAMES, actuator_joint_names, "joint stiffness")
    damping_actuator = kd_scale * _reorder(G1_DAMPING, G1_JOINT_NAMES, actuator_joint_names, "joint damping")
    steps_per_motion_frame = max(1, int(round(motion_dt / physics_dt)))
    render_decimation = render_decimation or steps_per_motion_frame

    print(f"[INFO]: Motion file: {motion_path}")
    print(f"[INFO]: XML path: {xml_path}")
    print(f"[INFO]: Mode: {mode}")
    print(f"[INFO]: Frames: {num_frames}, fps: {fps:g}, physics_dt: {physics_dt:g}")
    print(f"[INFO]: Root: write_root={write_root}, init_root={init_root}, pin_root={pin_root}")
    print(f"[INFO]: PD gains: kp_scale={kp_scale:g}, kd_scale={kd_scale:g}")
    print(f"[INFO]: MuJoCo actuator joint order: {actuator_joint_names}")

    if speed <= 0:
        raise ValueError("--speed must be positive.")
    if kp_scale <= 0 or kd_scale <= 0:
        raise ValueError("--kp_scale and --kd_scale must be positive.")

    if mode == "kinematic":
        total_frames = max_frames
        if total_frames is None:
            total_frames = num_frames

        def run_kinematic(viewer=None):
            frame_counter = 0
            last_idx = 0
            while total_frames is None or frame_counter < total_frames:
                frame_start = time.time()
                last_idx = _kinematic_step(
                    model,
                    data,
                    motion,
                    frame_counter,
                    num_frames,
                    actuator_joint_names,
                    loop,
                    write_root,
                    root_height_offset,
                )
                if viewer is not None:
                    viewer.sync()
                frame_counter += 1
                if not loop and frame_counter >= num_frames and total_frames is None:
                    break
                sleep_time = motion_dt / speed - (time.time() - frame_start)
                if sleep_time > 0 and not headless:
                    time.sleep(sleep_time)
            print(f"[INFO]: Kinematic replay completed: frames={frame_counter}, last_motion_frame={last_idx}")

        if headless:
            run_kinematic()
            return
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_kinematic(viewer)
        return

    if mode == "pd":
        write_motion_frame_to_mujoco(
            model,
            data,
            motion,
            0,
            actuator_joint_names,
            write_root=init_root,
            root_height_offset=root_height_offset,
        )
        total_steps = max_steps
        if total_steps is None:
            frames_to_run = max_frames if max_frames is not None else num_frames
            total_steps = frames_to_run * steps_per_motion_frame

        def run_pd(viewer=None):
            last_idx = 0
            for step_counter in range(total_steps):
                step_start = time.time()
                last_idx = _pd_step(
                    model,
                    data,
                    motion,
                    step_counter,
                    num_frames,
                    steps_per_motion_frame,
                    actuator_joint_names,
                    stiffness_actuator,
                    damping_actuator,
                    loop,
                    pin_root,
                    root_height_offset,
                )
                if viewer is not None and step_counter % render_decimation == 0:
                    viewer.sync()
                sleep_time = physics_dt / speed - (time.time() - step_start)
                if sleep_time > 0 and not headless:
                    time.sleep(sleep_time)
            print(f"[INFO]: PD replay completed: steps={total_steps}, last_motion_frame={last_idx}")

        if headless:
            run_pd()
            return
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_pd(viewer)
        return

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a G1 motion NPZ directly in MuJoCo.")
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
        default=str(_first_existing_path(DEFAULT_G1_XML_CANDIDATES)),
        help="Path to the G1 MuJoCo XML file.",
    )
    parser.add_argument("--mode", choices=["kinematic", "pd"], default="kinematic", help="Replay mode.")
    parser.add_argument("--loop", action="store_true", help="Loop motion when reaching the end.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a MuJoCo viewer.")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum motion frames to replay.")
    parser.add_argument("--max_steps", type=int, default=None, help="Maximum MuJoCo steps for --mode pd.")
    parser.add_argument("--physics_dt", type=float, default=DEFAULT_PHYSICS_DT, help="MuJoCo physics timestep.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier.")
    parser.add_argument(
        "--no_root",
        action="store_true",
        help="Do not initialize/write the floating base from motion.",
    )
    parser.add_argument(
        "--pin_root",
        action="store_true",
        help="In --mode pd, overwrite the floating base from motion at every step. This is not physical.",
    )
    parser.add_argument(
        "--root_height_offset",
        type=float,
        default=0.0,
        help="Add a z offset to motion root height when root state is written.",
    )
    parser.add_argument("--kp_scale", type=float, default=1.0, help="PD stiffness multiplier for --mode pd.")
    parser.add_argument("--kd_scale", type=float, default=1.0, help="PD damping multiplier for --mode pd.")
    parser.add_argument(
        "--render_decimation",
        type=int,
        default=CONTROL_DECIMATION,
        help="Viewer sync decimation for --mode pd.",
    )
    args = parser.parse_args()

    motion_file = args.motion_file
    if Path(motion_file).expanduser().is_dir():
        motion_file = _resolve_motion_file(motion_file, args.motion_name)

    run_motion2sim(
        motion_file,
        args.xml_path,
        mode=args.mode,
        loop=args.loop,
        headless=args.headless,
        max_frames=args.max_frames,
        max_steps=args.max_steps,
        physics_dt=args.physics_dt,
        speed=args.speed,
        write_root=not args.no_root,
        init_root=not args.no_root,
        pin_root=args.pin_root,
        root_height_offset=args.root_height_offset,
        kp_scale=args.kp_scale,
        kd_scale=args.kd_scale,
        render_decimation=args.render_decimation,
    )


if __name__ == "__main__":
    main()
