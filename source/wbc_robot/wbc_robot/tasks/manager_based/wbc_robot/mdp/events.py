from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    随机化关节的默认位置，由于标定误差，这些位置可能与 URDF 中的不同。
    """
    # 提取使用的变量 (以启用类型提示)
    asset: Articulation = env.scene[asset_cfg.name]

    # 导出时保存标称值
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # 解析环境 ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # 解析关节索引
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # 出于优化目的
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # 更新 action 中的 offset，因为它不会自动更新
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """通过加入从给定范围采样得到的随机值来随机化刚体的质心 (CoM)。

    .. note::
        此函数使用 CPU 张量来赋值 CoM。建议仅在初始化环境时使用此函数。
    """
    # 提取使用的变量 (以启用类型提示)
    asset: Articulation = env.scene[asset_cfg.name]
    # 解析环境 ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # 解析刚体索引
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # 采样随机质心(CoM)值
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # 获取当前刚体的质心 (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # 在指定范围内随机化质心
    coms[:, body_ids, :3] += rand_samples

    # 设置新质心
    asset.root_physx_view.set_coms(coms, env_ids)
