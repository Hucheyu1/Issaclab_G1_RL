# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Play one GAEMimic motion from frame 0 to the last frame, then stop."""

"""Launch Isaac Sim Simulator first."""

import argparse
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate one GAEMimic motion clip with an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during evaluation.")
parser.add_argument("--video_length", type=int, default=None, help="Recorded video length in policy steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the GAEMimic task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--motion_file", type=str, required=True, help="Path to the single GAEMimic npz motion file.")
parser.add_argument(
    "--activate_signals",
    type=str,
    default=None,
    choices=["robot", "smplx", "keypoints"],
    help="Override the active signals modality during testing.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional cap on policy steps. Defaults to the number of frames in --motion_file.",
)
parser.add_argument(
    "--progress_interval",
    type=int,
    default=100,
    help="Print progress every N policy steps. Set to 0 to disable.",
)
parser.add_argument(
    "--allow_early_resets",
    action="store_true",
    default=False,
    help="Keep normal termination checks. By default they are disabled for full-clip evaluation.",
)
parser.add_argument(
    "--tmp_dataset_root",
    type=str,
    default="/root/gpufree-data/lab_lecture/wbc_robot/datasets/wbc_robot_paly_one",
    help="Directory used to build a temporary one-motion dataset.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import time
import types
from collections.abc import Sequence

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
import wbc_robot.tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner  # type: ignore


def _motion_metadata(motion_file: str) -> tuple[Path, int, float]:
    """Return canonical path, frame count, and fps for a motion npz."""
    motion_path = Path(motion_file).expanduser().resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(f"Motion file does not exist: {motion_path}")

    with np.load(motion_path) as data:
        if "joint_pos" not in data:
            raise KeyError(f"{motion_path} is missing required key: joint_pos")
        frame_count = int(data["joint_pos"].shape[0])
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0

    if frame_count <= 0:
        raise ValueError(f"{motion_path} contains no frames.")
    if fps <= 0:
        raise ValueError(f"{motion_path} has invalid fps: {fps}")

    return motion_path, frame_count, fps


def _safe_motion_name(motion_path: Path) -> str:
    """Create a dataset-safe motion name from the source file name."""
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", motion_path.stem).strip("._")
    digest = hashlib.sha1(str(motion_path).encode("utf-8")).hexdigest()[:8]
    return f"{safe_stem or 'motion'}_{digest}"


def _build_single_motion_dataset(motion_path: Path, tmp_dataset_root: str) -> str:
    """Build a minimal Motion_Dataset-compatible directory for exactly one npz."""
    motion_name = _safe_motion_name(motion_path)
    dataset_dir = Path(tmp_dataset_root).expanduser().resolve() / motion_name
    robot_dir = dataset_dir / "g1"
    robot_dir.mkdir(parents=True, exist_ok=True)

    target_motion = robot_dir / f"{motion_name}.npz"
    if target_motion.exists() or target_motion.is_symlink():
        target_motion.unlink()
    try:
        os.symlink(motion_path, target_motion)
    except OSError:
        shutil.copy2(motion_path, target_motion)

    info_yaml = dataset_dir / "info.yaml"
    info_yaml.write_text(
        f'dataset: "single_motion_eval"\neval:\n  {motion_name}: 1\n',
        encoding="utf-8",
    )
    return str(dataset_dir)


def _zero_motion_reset_noise(motion_cfg):
    """Disable random pose, velocity, and joint reset noise for the motion command."""
    motion_cfg.pose_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    motion_cfg.velocity_range = {key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.adaptive_uniform_ratio = 0.0
    motion_cfg.adaptive_cap = 1


def _disable_early_terminations(env_cfg):
    """Remove tracking-failure termination terms so the clip can finish."""
    terminations = getattr(env_cfg, "terminations", None)
    if terminations is None:
        return
    for name in ("anchor_pos", "anchor_ori", "ee_body_pos"):
        if hasattr(terminations, name):
            setattr(terminations, name, None)


def _patch_motion_command_for_eval(command):
    """Make GAEMimic command resampling deterministic and non-random."""

    def deterministic_resample(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if isinstance(env_ids, torch.Tensor):
            env_ids_tensor = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)

        self.motion_ids[env_ids_tensor] = 0
        last_step = self.dataloader.motion_lengths[0] - 1
        local_steps = torch.clamp(self.time_steps[env_ids_tensor], min=0, max=last_step)
        self.time_steps[env_ids_tensor] = local_steps
        self.global_time_steps[env_ids_tensor] = self.dataloader.motion_offsets[0] + local_steps

        root_pos = self.body_pos_w[env_ids_tensor, 0].clone()
        root_ori = self.body_quat_w[env_ids_tensor, 0].clone()
        root_lin_vel = self.body_lin_vel_w[env_ids_tensor, 0].clone()
        root_ang_vel = self.body_ang_vel_w[env_ids_tensor, 0].clone()
        joint_pos = self.joint_pos[env_ids_tensor].clone()
        joint_vel = self.joint_vel[env_ids_tensor].clone()

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_tensor)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids_tensor,
        )

    command._resample_command = types.MethodType(deterministic_resample, command)


def _force_motion_frame_zero(raw_env):
    """Set the command and robot state to frame 0 after wrapper resets."""
    command = raw_env.command_manager.get_term("motion")
    command.motion_ids[:] = 0
    command.time_steps[:] = 0
    command.global_time_steps[:] = command.dataloader.motion_offsets[0]

    env_ids = torch.arange(raw_env.num_envs, dtype=torch.long, device=command.device)
    root_state = torch.cat(
        [
            command.body_pos_w[:, 0].clone(),
            command.body_quat_w[:, 0].clone(),
            command.body_lin_vel_w[:, 0].clone(),
            command.body_ang_vel_w[:, 0].clone(),
        ],
        dim=-1,
    )
    command.robot.write_joint_state_to_sim(command.joint_pos.clone(), command.joint_vel.clone(), env_ids=env_ids)
    command.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    raw_env.scene.write_data_to_sim()
    return command


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Evaluate one GAEMimic motion with RSL-RL."""
    motion_path, motion_frames, motion_fps = _motion_metadata(args_cli.motion_file)
    single_motion_dataset = _build_single_motion_dataset(motion_path, args_cli.tmp_dataset_root)

    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env_cfg.commands.motion.dataset_dirs = [single_motion_dataset]
    env_cfg.commands.motion.robot_name = "g1"
    env_cfg.commands.motion.splits = ["eval"]
    _zero_motion_reset_noise(env_cfg.commands.motion)

    if not args_cli.allow_early_resets:
        _disable_early_terminations(env_cfg)

    policy_dt = env_cfg.decimation * env_cfg.sim.dt
    eval_steps = args_cli.max_steps if args_cli.max_steps is not None else motion_frames
    eval_steps = max(1, min(eval_steps, motion_frames))
    env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), (eval_steps + 2) * policy_dt)

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    print("[INFO] Single-motion GAEMimic evaluation:")
    print(f"  - motion_file: {motion_path}")
    print(f"  - temp_dataset: {single_motion_dataset}")
    print(f"  - frames/fps: {motion_frames}/{motion_fps:g}")
    print(f"  - eval_steps: {eval_steps}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    raw_env = env.unwrapped
    motion_command = raw_env.command_manager.get_term("motion")
    _patch_motion_command_for_eval(motion_command)

    if args_cli.video:
        video_length = args_cli.video_length if args_cli.video_length is not None else eval_steps
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "paly_one"),
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during single-motion evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    _force_motion_frame_zero(env.unwrapped)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if hasattr(agent_cfg.policy, "activate_signals") and args_cli.activate_signals is not None:
        agent_cfg.policy.activate_signals = args_cli.activate_signals
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs, _ = env.get_observations()

    for step in range(eval_steps):
        if not simulation_app.is_running():
            break

        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        if args_cli.progress_interval and (
            step == 0 or (step + 1) % args_cli.progress_interval == 0 or step + 1 == eval_steps
        ):
            current_frame = int(env.unwrapped.command_manager.get_term("motion").time_steps[0].item())
            print(f"[INFO] step {step + 1}/{eval_steps}, command_frame={current_frame}/{motion_frames - 1}")

        if torch.any(dones):
            print("[WARN] Environment reset occurred during evaluation. Use default settings to disable early resets.")

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    print("[INFO] Reached the requested end of the single-motion evaluation.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
