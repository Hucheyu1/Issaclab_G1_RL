from __future__ import annotations

from dataclasses import MISSING
from typing import Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl.rl_cfg import RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRl_Triple_AE_PPOPolicyCfg(RslRlPpoActorCriticCfg):
    """Triple_AE_PPO策略的配置。"""

    class_name: str = "ActorCritic_Triple_AE"
    """策略类名。默认为ActorCritic_Triple_AE。"""

    actor_sg_dim: int = MISSING
    """策略网络(Actor)的机器人状态维度。"""

    actor_sh_dim: int = MISSING
    """策略网络(Actor)的人类状态维度。"""

    actor_sk_dim: int = MISSING
    """策略网络(Actor)的关键点SE3状态维度。"""

    latent_dim: int = MISSING
    """三重自编码器(Triple Autoencoder)的潜在/隐空间维度。"""

    activate_signals: Literal["robot", "smplx", "keypoints"] = "robot"
    """要激活的信号：'robot'(机器人)、'smplx'(人体模型) 或 'keypoints'(关键点)。默认为'robot'。"""

    robot_encoder_hidden_dims: list[int] = MISSING
    """机器人编码器网络的隐藏层维度。"""

    human_encoder_hidden_dims: list[int] = MISSING
    """人类编码器网络的隐藏层维度。"""

    keypoints_encoder_hidden_dims: list[int] = MISSING
    """关键点编码器网络的隐藏层维度。"""

    robot_decoder_hidden_dims: list[int] = MISSING
    """机器人解码器网络的隐藏层维度。"""

    human_decoder_hidden_dims: list[int] = MISSING
    """人类解码器网络的隐藏层维度。"""

    keypoints_decoder_hidden_dims: list[int] = MISSING
    """关键点解码器网络的隐藏层维度。"""


@configclass
class RslRl_Triple_AE_PPOAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Triple_AE_PPO算法的配置。"""

    class_name: str = "Triple_AE_PPO"
    """算法类名。默认为Triple_AE_PPO。"""

    reconstruction_loss_coef_sg: float = MISSING
    """机器人目标状态重建损失的系数。"""

    reconstruction_loss_coef_sh: float = MISSING
    """人类状态重建损失的系数。"""

    reconstruction_loss_coef_sk: float = MISSING
    """关键点状态重建损失的系数。"""

    alignment_loss_coef: float = MISSING
    """三路潜空间特征对齐损失(MSE)的系数。"""

    consistency_loss_coef: float = MISSING
    """跨模态一致性损失的系数。"""

    finetune_human_encoder: bool = False
    """是否微调人类编码器和解码器。默认为False。"""

    finetune_robot_encoder: bool = False
    """是否微调机器人编码器和解码器。默认为False。"""

    finetune_keypoints_encoder: bool = False
    """是否微调关键点编码器和解码器。默认为False。"""


@configclass
class RslRl_Triple_AE_PPO_Single_Finetune_PolicyCfg(RslRlPpoActorCriticCfg):
    """Triple_AE_PPO_Single_Finetune单模态微调策略的配置。

    该策略专为单模态数据的微调而设计，并且其编码器/解码器是冻结的。
    默认情况下，指令(cmd)的编码器和解码器被冻结，仅训练动作(action)解码器。
    """

    class_name: str = "ActorCritic_Triple_AE_Single_Finetune"
    """策略类名。默认为ActorCritic_Triple_AE_Single_Finetune。"""

    actor_cmd_dim: int = MISSING
    """策略网络(Actor)的指令(cmd)状态维度（即输入到编码器的数据维度）。"""

    latent_dim: int = 32
    """自编码器的潜在/隐空间维度。默认为32。"""

    cmd_encoder_hidden_dims: list[int] = MISSING
    """指令(cmd)编码器网络的隐藏层维度。"""

    cmd_decoder_hidden_dims: list[int] = MISSING
    """指令(cmd)解码器网络的隐藏层维度。"""

    freeze: bool = True
    """是否冻结指令(cmd)的编码器和解码器。默认为True。"""


@configclass
class RslRl_Triple_AE_PPO_Single_Finetune_AlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Triple_AE_PPO_Single_Finetune单模态微调算法的配置。

    该算法用于在单一模态上微调预先训练好的Triple_AE网络模型。
    这里仅应用指令(cmd)状态的重建损失（不会应用对齐或一致性损失）。
    """

    class_name: str = "Triple_AE_PPO_Single_Finetune"
    """算法类名。默认为Triple_AE_PPO_Single_Finetune。"""

    reconstruction_loss_coef_cmd: float = 0.0
    """指令(cmd)状态重建损失的系数。默认为0.0。"""
