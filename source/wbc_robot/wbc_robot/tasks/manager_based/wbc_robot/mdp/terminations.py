from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from wbc_robot.tasks.manager_based.wbc_robot.mdp.commands import MotionCommand
from wbc_robot.tasks.manager_based.wbc_robot.mdp.rewards import _get_body_indexes


# ---------------------------------------------------------
# 1. 锚点三维位置容差判定 (Anchor Position Error)
# 作用：计算指令(目标)和机器人实际锚点（通常是骨盆/Root节点）
# 在3D空间内的欧氏距离，如果距离大于阈值 threshold 则判定当前回合失败(摔倒或脱节)。
# ---------------------------------------------------------
def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


# ---------------------------------------------------------
# 2. 锚点高度容差判定 (Anchor Z-axis Error)
# 作用：只对比锚点的 Z 轴（高度）误差。
# 通常用于机器人高度偏离（比如重重摔地、或者跳离目标过高）时提前终止。
# ---------------------------------------------------------
def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


# ---------------------------------------------------------
# 3. 锚点朝向容差判定 (Anchor Orientation Error)
# 作用：判断机器人躯干朝向是否离谱脱轨。
# 原理计算：分别将真实世界的重力向量 (0,0,-9.8) 根据目标四元数和机器人实际四元数，
# 反投影回锚点的局部坐标系中。如果两者局部感受到的 Z 轴朝下重力分量相差超过设定的阈值，
# 说明机器人的倾斜状态与动捕参考相差悬殊，判定失败。
# ---------------------------------------------------------
def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


# ---------------------------------------------------------
# 4. 指定刚体部位三维位置容差判定 (Specific Body Position Error)
# 作用：获取被要求追踪身体部位相对锚点的世界坐标。
# 计算机器人相关部位与目标轨迹部位部位的 3D 距离差。
# 只要指定的部位中有任何一个误差大于 threshold，就触发终止信号。
# ---------------------------------------------------------
def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


# ---------------------------------------------------------
# 5. 指定刚体部位高度容差判定 (Specific Body Z-axis Error)
# 作用：仅检测某些关注部位在 Z 轴高度上的误差。
# 这种机制对诸如脚部踩地情况尤为敏锐，避免脚悬空过高但其他轴都没超时没有被捕获终止。
# ---------------------------------------------------------
def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
