from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# 它通过观察机器人在最近一个时间段内的历史表现（通过 Reward 奖励值来衡量），动态调整 motion（动作指令生成）中的 adaptive_uniform_ratio (均匀采样与自适应采样的混合比例)。
# 换句话说：如果机器人最近表现变好了，就增加随机/均匀抽样的动作让它训练更多样化；如果还没学好，就不增加，让算法继续多练它不擅长的动作（自适应采样）。
def adaptive_sampling_ratio(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    max_ratio: float = 0.9,
    delta_ratio: float = 2e-3,
    threshold: float = 0.9,
    episode_num: int = 1,
) -> torch.Tensor:
    # use episode alive length to adjust the sampling ratio between uniform and adaptive sampling
    command_term = env.command_manager.get_term("motion")

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    if env.common_step_counter % (env.max_episode_length * episode_num) == 0:
        if reward > reward_term.weight * threshold:
            command_term.cfg.adaptive_uniform_ratio = min(
                max_ratio, command_term.cfg.adaptive_uniform_ratio + delta_ratio
            )

    return torch.tensor(command_term.cfg.adaptive_uniform_ratio, device=env.device)
