from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from wbc_robot.tasks.manager_based.wbc_robot.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

# 获取机器人的本体骨盆中心（Anchor）在世界坐标系(_w)下的朝向。
# 计算： 它通过 matrix_from_quat 把四元数转成了3×3 旋转矩阵，但注意切片 mat[..., :2]。在强化学习中，表示 3D 旋转用完整的 9 个元素的旋转矩阵冗余，
# 用四元数又因为存在“双重覆盖（q和-q是同一姿态）”易造成神经网络发散。所以这里采用了机器人领域经典的 6D朝向表示法：取旋转矩阵的前两列（6个数字），
# 这 6 个数字可以被网络无歧义且连续地解读。最后展开拉平 (reshape)。
def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)

# 把机器人的线速度和角速度获取过来，直接切片取出线速度的前3维和角速度的后3维，并拉平展开。
def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)

# 获取机器人全身刚体在局部坐标系中的位置姿态
# 用途： 神经网络只靠关节角度很难直观感知自己四肢究竟在哪，因此这个函数给 Critic（评论家网络）提供机器人全身骨架的精确定位
# 计算： subtract_frame_transforms 相当于执行了一个坐标系变换。它把 14个全身连杆（大腿小腿等）在世界坐标系(_w)的位置robot_body_pos_w，减去了骨盆中心的位置robot_anchor_pos_w。
# 结果： 算出来的是“左手相对于肚子在哪里”、“右脚相对于肚子在哪里”，这就叫做 Body 系下的坐标(_b)。最后拉平成所有连杆的位置张量传出。
def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)

# 用途： 与上一个函数相仿，这负责求解各个部分连杆相对于主躯干的旋转差（朝向）。并也是提取前两列转成了 6D 连续姿态向量（14部件 × 6维 = 84维度数据）。
def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

# 获取目标参考姿态(Command)相对于机器人的差值
def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

# 从数据集中提取一段未来的机器人关节目标轨迹
# frames=10, interval=2，它获取的不仅是当前帧的目标，还会每隔 2 步往后看，一共拿 10 帧的数据
# 设你的控制步长是 0.02秒，间隔为 2 意味着每 0.04 秒采一帧，一共采 10 帧。这意味着网络不仅能看到当前时刻的目标，还能“预见”未来 0.4 秒的完整动作轨迹
def motion_robot_joint_pos(
    env: ManagerBasedEnv,
    command_name: str,
    interval: int,
    frames: int,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)

    return command.motion_robot_joint_pos(interval, frames).view(env.num_envs, -1)


def motion_robot_joint_pos_vel(
    env: ManagerBasedEnv,
    command_name: str,
    interval: int,
    frames: int,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)

    return command.motion_robot_joint_pos_vel(interval, frames).view(env.num_envs, -1)


def motion_smplx_pose_body(
    env: ManagerBasedEnv,
    command_name: str,
    interval: int,
    frames: int,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)

    return command.motion_smplx_pose_body(interval, frames).view(env.num_envs, -1)


def motion_keypoints_se3(
    env: ManagerBasedEnv,
    command_name: str,
    interval: int,
    frames: int,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)

    return command.motion_keypoints_se3(interval, frames).view(env.num_envs, -1)
