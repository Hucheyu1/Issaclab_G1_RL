import torch
import torch.nn as nn
from typing import Literal
from torch.distributions import Normal
import re

from rsl_rl.utils import resolve_nn_activation
from rsl_rl.modules.normalizer import EmpiricalNormalization


class Actor_Triple_AE(nn.Module):
    """
    多模态机器人控制与关键点学习的三重自编码器(Triple Autoencoder)。
    
    核心网络组件：
    - 机器人编码器-解码器对：用于编码/解码机器人目标状态 (x_sg)
    - 人类编码器-解码器对：用于编码/解码人类状态 (x_sh)
    - 关键点编码器-解码器对：用于编码/解码关键点 SE3 变换 (x_sk)
    - 动作解码器：基于隐层向量 + 本体状态(proprioceptive state) 生成策略动作
    
    与 Dual_AE 架构的主要区别：
    1. 为三种模态（机器人、人类、关键点）引入了三个独立的编码器
    2. 为重建过程保留三个独立的解码器
    3. 在共享隐式特征空间中进行三向对齐损失 (Three-way alignment loss) 计算
    4. 针对所有三种模态增加了跨模态一致性损失验证 (Cross-modal consistency loss)
    
    主要信号流 (当 activate_signals="robot" 时):
    
    x_sg (机器人状态) --robot_encoder--> z_robot --robot_decoder--> x_sg_recon (状态重建)
                                              |
                                              ├--[与 x_sp 拼接]--action_decoder--> actions (生成动作)
                                              |
    x_sh (人类状态)   --human_encoder--> z_human --human_decoder--> x_sh_recon (状态重建)
                                              |
    x_sk (关键点状态) --keypoints_encoder--> z_keypoints --keypoints_decoder--> x_sk_recon (状态重建)
                                              |
    x_sp (本体状态)  --> EmpiricalNormalization --> concat 沿特征维拼接到隐变量 z
    
    损失函数组件（在外部通过计算介入 RL 算法训练环节）：
    - 对齐(alignment)：三向 MSE 对齐误差评估 (z_sg 与 z_sh, z_sg 与 z_sk, z_sh 与 z_sk 之间)
    - 重构(reconstruction)：对于三种重建输出的 MSE 评估 (x_sg, x_sh, x_sk)
    - 一致性(consistency)：基于激活信号 `activate_signals` 模态选择进行的交叉重建一致性映射损失
    """
    
    def __init__(
        self,
        # In-Out Dimensions
        num_actor_obs: int,
        num_actions: int,
        actor_sg_dim: int,           # robot state dimension
        actor_sh_dim: int = 0,       # human state dimension
        actor_sk_dim: int = 0,       # keypoints state dimension
        # AE Configuration
        latent_dim: int = 32,
        # Network Architecture
        robot_encoder_hidden_dims: list[int] = None,
        human_encoder_hidden_dims: list[int] = None,
        keypoints_encoder_hidden_dims: list[int] = None,
        robot_decoder_hidden_dims: list[int] = None,
        human_decoder_hidden_dims: list[int] = None,
        keypoints_decoder_hidden_dims: list[int] = None,
        action_decoder_hidden_dims: list[int] = None,
        activation: str = "relu",
        # Modality routing
        activate_signals: Literal["robot", "smplx", "keypoints"] = "robot",
    ):
        super().__init__()
        
        self.num_actor_obs = num_actor_obs
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.actor_sk_dim = actor_sk_dim
        self.num_actions = num_actions
        self.activate_signals = activate_signals
        self.latent_dim = latent_dim
        
        # Set default hidden dimensions if not provided
        if robot_encoder_hidden_dims is None:
            robot_encoder_hidden_dims = [512, 256]
        if human_encoder_hidden_dims is None:
            human_encoder_hidden_dims = [512, 256]
        if keypoints_encoder_hidden_dims is None:
            keypoints_encoder_hidden_dims = [512, 256]
        if robot_decoder_hidden_dims is None:
            robot_decoder_hidden_dims = [256, 512]
        if human_decoder_hidden_dims is None:
            human_decoder_hidden_dims = [256, 512]
        if keypoints_decoder_hidden_dims is None:
            keypoints_decoder_hidden_dims = [256, 512]
        
        activation_fn = resolve_nn_activation(activation)
        
        # ==================== Robot Encoder ====================
        # Input: x_sg (actor_sg_dim) -> Output: latent_dim
        robot_encoder_layers = []
        robot_encoder_layers.append(nn.Linear(actor_sg_dim, robot_encoder_hidden_dims[0]))
        robot_encoder_layers.append(activation_fn)
        for i in range(len(robot_encoder_hidden_dims) - 1):
            robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[i], robot_encoder_hidden_dims[i+1]))
            robot_encoder_layers.append(activation_fn)
        robot_encoder_layers.append(nn.Linear(robot_encoder_hidden_dims[-1], latent_dim))
        self.robot_encoder = nn.Sequential(*robot_encoder_layers)
        
        # ==================== Human Encoder ====================
        # Input: x_sh (actor_sh_dim) -> Output: latent_dim
        human_encoder_layers = []
        human_encoder_layers.append(nn.Linear(actor_sh_dim, human_encoder_hidden_dims[0]))
        human_encoder_layers.append(activation_fn)
        for i in range(len(human_encoder_hidden_dims) - 1):
            human_encoder_layers.append(nn.Linear(human_encoder_hidden_dims[i], human_encoder_hidden_dims[i+1]))
            human_encoder_layers.append(activation_fn)
        human_encoder_layers.append(nn.Linear(human_encoder_hidden_dims[-1], latent_dim))
        self.human_encoder = nn.Sequential(*human_encoder_layers)
        
        # ==================== Keypoints Encoder ====================
        # Input: x_sk (actor_sk_dim) -> Output: latent_dim
        keypoints_encoder_layers = []
        keypoints_encoder_layers.append(nn.Linear(actor_sk_dim, keypoints_encoder_hidden_dims[0]))
        keypoints_encoder_layers.append(activation_fn)
        for i in range(len(keypoints_encoder_hidden_dims) - 1):
            keypoints_encoder_layers.append(nn.Linear(keypoints_encoder_hidden_dims[i], keypoints_encoder_hidden_dims[i+1]))
            keypoints_encoder_layers.append(activation_fn)
        keypoints_encoder_layers.append(nn.Linear(keypoints_encoder_hidden_dims[-1], latent_dim))
        self.keypoints_encoder = nn.Sequential(*keypoints_encoder_layers)
        
        # ==================== Robot Decoder ====================
        # Input: latent_dim -> Output: x_sg (actor_sg_dim)
        robot_decoder_layers = []
        robot_decoder_layers.append(nn.Linear(latent_dim, robot_decoder_hidden_dims[0]))
        robot_decoder_layers.append(activation_fn)
        for i in range(len(robot_decoder_hidden_dims) - 1):
            robot_decoder_layers.append(nn.Linear(robot_decoder_hidden_dims[i], robot_decoder_hidden_dims[i+1]))
            robot_decoder_layers.append(activation_fn)
        robot_decoder_layers.append(nn.Linear(robot_decoder_hidden_dims[-1], actor_sg_dim))
        self.robot_decoder = nn.Sequential(*robot_decoder_layers)
        
        # ==================== Human Decoder ====================
        # Input: latent_dim -> Output: x_sh (actor_sh_dim)
        human_decoder_layers = []
        human_decoder_layers.append(nn.Linear(latent_dim, human_decoder_hidden_dims[0]))
        human_decoder_layers.append(activation_fn)
        for i in range(len(human_decoder_hidden_dims) - 1):
            human_decoder_layers.append(nn.Linear(human_decoder_hidden_dims[i], human_decoder_hidden_dims[i+1]))
            human_decoder_layers.append(activation_fn)
        human_decoder_layers.append(nn.Linear(human_decoder_hidden_dims[-1], actor_sh_dim))
        self.human_decoder = nn.Sequential(*human_decoder_layers)
        
        # ==================== Keypoints Decoder ====================
        # Input: latent_dim -> Output: x_sk (actor_sk_dim)
        keypoints_decoder_layers = []
        keypoints_decoder_layers.append(nn.Linear(latent_dim, keypoints_decoder_hidden_dims[0]))
        keypoints_decoder_layers.append(activation_fn)
        for i in range(len(keypoints_decoder_hidden_dims) - 1):
            keypoints_decoder_layers.append(nn.Linear(keypoints_decoder_hidden_dims[i], keypoints_decoder_hidden_dims[i+1]))
            keypoints_decoder_layers.append(activation_fn)
        keypoints_decoder_layers.append(nn.Linear(keypoints_decoder_hidden_dims[-1], actor_sk_dim))
        self.keypoints_decoder = nn.Sequential(*keypoints_decoder_layers)
        
        # ==================== Proprioceptive State Normalizer ====================
        actor_sp_dim = num_actor_obs - actor_sg_dim - actor_sh_dim - actor_sk_dim
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(actor_sp_dim,))
        
        # ==================== Action Decoder ====================
        # Input: latent_dim + actor_sp_dim -> Output: num_actions
        action_layers = []
        action_layers.append(nn.Linear(latent_dim + actor_sp_dim, action_decoder_hidden_dims[0]))
        action_layers.append(activation_fn)
        for i in range(len(action_decoder_hidden_dims) - 1):
            action_layers.append(nn.Linear(action_decoder_hidden_dims[i], action_decoder_hidden_dims[i+1]))
            action_layers.append(activation_fn)
        action_layers.append(nn.Linear(action_decoder_hidden_dims[-1], num_actions))
        self.action_decoder = nn.Sequential(*action_layers)
    
    def forward(self, x):
        """
        通过 Triple_AE Actor 网络的前向传播（附带模态路由功能）。
        
        基于 `activate_signals` 的指示，使用确定性编码器将对应的状态转化为隐层特征。
        
        参数:
            x: 形状为 (batch_size, num_actor_obs) 的拼接输入张量
               数据结构通常为: [x_sg (机器人) | x_sh (人类) | x_sk (关键点) | x_sp (自身本体维度)]
        
        返回:
            actions: 预测输出的动作, 形状为 (batch_size, num_actions)
        """
        if self.activate_signals == "robot":
            return self.forward_robot(x)
        elif self.activate_signals == "smplx":
            return self.forward_smplx(x)
        elif self.activate_signals == "keypoints":
            return self.forward_keypoints(x)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}. Must be 'robot', 'smplx', or 'keypoints'.")
    
    def forward_robot(self, x):
        """使用【机器人自身目标状态】作为核心输入模态的前向传播过程"""
        # 切分重组拼接张量，分离出各项：[x_sg | x_sh | x_sk | x_sp]
        x_sg = x[:, :self.actor_sg_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        # 将机器人状态编码为隐变量特征(z)，这是确定性映射（不包含 VAE 的采样)
        z_sg = self.robot_encoder(x_sg)
        
        # 分离(detach)隐层张量，防止 PPO 回传的策略梯度直接穿透到编码器(Encoder)内部造成干涉
        # 这些编码网络应当由外挂的辅助损失（重构、对齐和一致性）单独计算并收敛
        z_for_action = z_sg.detach()
        
        # 标准化本体传感状态
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # 向量拼接后经过解码器输出具体的执行 Action
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_smplx(self, x):
        """使用【SMPLX人类状态】作为核心输入模态的前向传播过程"""
        # 切分重组拼接张量，分离出各项：[x_sg | x_sh | x_sk | x_sp]
        x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        # Encode human state to latent (deterministic, no sampling)
        z_sh = self.human_encoder(x_sh)
        
        # 分离(detach)隐层张量，防止 PPO 回传的策略梯度直接穿透到编码器(Encoder)内部造成干涉
        z_for_action = z_sh.detach()
        
        # 标准化本体传感状态
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # 向量拼接后经过解码器输出具体的执行 Action
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def forward_keypoints(self, x):
        """使用【关键点 SE3 变换】作为核心输入模态的前向传播过程"""
        # 切分重组拼接张量，分离出各项：[x_sg | x_sh | x_sk | x_sp]
        x_sk = x[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        x_sp = x[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        # Encode keypoints state to latent (deterministic, no sampling)
        z_sk = self.keypoints_encoder(x_sk)
        
        # 分离(detach)隐层张量，防止 PPO 回传的策略梯度直接穿透到编码器(Encoder)内部造成干涉
        z_for_action = z_sk.detach()
        
        # 标准化本体传感状态
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # 向量拼接后经过解码器输出具体的执行 Action
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def encode_robot(self, x):
        """使用机器人编码器进行特征编码 (确定性，不采样)。
        
        参数:
            x: 输入张量，形状为 (batch_size, num_actor_obs) 或 (batch_size, actor_sg_dim)
        
        返回:
            z: 隐层向量 (确定性计算得到，不含随机采样)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sg = x[:, :self.actor_sg_dim]
        else:
            x_sg = x
        
        z = self.robot_encoder(x_sg)
        return z
    
    def encode_smplx(self, x):
        """使用人类行为编码器进行特征编码 (确定性，不采样)。
        
        参数:
            x: 输入张量，形状为 (batch_size, num_actor_obs) 或 (batch_size, actor_sh_dim)
        
        返回:
            z: 隐层向量 (确定性计算得到，不含随机采样)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sh = x[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        else:
            x_sh = x
        
        z = self.human_encoder(x_sh)
        return z
    
    def encode_keypoints(self, x):
        """使用关键点编码器进行特征编码 (确定性，不采样)。
        
        参数:
            x: 输入张量，形状为 (batch_size, num_actor_obs) 或 (batch_size, actor_sk_dim)
        
        返回:
            z: 隐层向量 (确定性计算得到，不含随机采样)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_sk = x[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        else:
            x_sk = x
        
        z = self.keypoints_encoder(x_sk)
        return z
    
    def decode_robot(self, z):
        """将隐层向量解码以重构机器人状态。
        
        参数:
            z: 隐层向量，形状为 (batch_size, latent_dim)
        
        返回:
            x_sg_recon: 重构出的机器人状态，形状为 (batch_size, actor_sg_dim)
        """
        return self.robot_decoder(z)
    
    def decode_human(self, z):
        """将隐层向量解码以重构人类状态。
        
        参数:
            z: 隐层向量，形状为 (batch_size, latent_dim)
        
        返回:
            x_sh_recon: 重构出的人类状态，形状为 (batch_size, actor_sh_dim)
        """
        return self.human_decoder(z)
    
    def decode_keypoints(self, z):
        """将隐层向量解码以重构关键点状态。
        
        参数:
            z: 隐层向量，形状为 (batch_size, latent_dim)
        
        返回:
            x_sk_recon: 重构出的关键点状态，形状为 (batch_size, actor_sk_dim)
        """
        return self.keypoints_decoder(z)
    
    def decode_action(self, z, x_sp):
        """根据隐层特征及本体感知状态解码生成动作。
        
        参数:
            z: 隐层向量，形状为 (batch_size, latent_dim)
            x_sp: 本体感知状态，形状为 (batch_size, actor_sp_dim)
        
        返回:
            actions: 网络的动作预测输出，形状为 (batch_size, num_actions)
        """
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.action_decoder(torch.cat([z, x_sp_normalized], dim=-1))
    
    def forward_robot_exporter(self, robot_command, proprioceptive_state):
        """针对部署阶段解耦解复用输入的前向推理(机器人模态)。
        
        参数:
            robot_command: 形状为 (batch_size, actor_sg_dim) 的张量
            proprioceptive_state: 形状为 (batch_size, actor_sp_dim) 的张量
        
        返回:
            actions: 网络的动作预测输出，形状为 (batch_size, num_actions)
        """
        z_sg = self.robot_encoder(robot_command)
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_sg, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def forward_smplx_exporter(self, smplx_human_state, proprioceptive_state):
        """针对部署阶段解耦解复用输入的前向推理(SMPLX模态)。
        
        参数:
            smplx_human_state: 形状为 (batch_size, actor_sh_dim) 的张量
            proprioceptive_state: 形状为 (batch_size, actor_sp_dim) 的张量
        
        返回:
            actions: 网络的动作预测输出，形状为 (batch_size, num_actions)
        """
        z_sh = self.human_encoder(smplx_human_state)
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_sh, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def forward_keypoints_exporter(self, keypoints_state, proprioceptive_state):
        """针对部署阶段解耦解复用输入的前向推理(关键点模态)。
        
        参数:
            keypoints_state: 形状为 (batch_size, actor_sk_dim) 的张量
            proprioceptive_state: 形状为 (batch_size, actor_sp_dim) 的张量
        
        返回:
            actions: 网络的动作预测输出，形状为 (batch_size, num_actions)
        """
        z_sk = self.keypoints_encoder(keypoints_state)
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_sk, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def freeze_cmd_encoder(self):
        """
        冻结所有的自编码器与状态解码器，仅保留“动作解码器”可供训练。
        
        用于针对特定下游微调任务(finetuning)。该方法将禁止所有三套（机器人、人类、关键点）
        输入层编码解构部分的梯度回传，只更新策略输出映射网络（Action Decoder）。
        """
        # List of all encoders and decoders
        encoders_decoders = [
            self.robot_encoder, self.human_encoder, self.keypoints_encoder,
            self.robot_decoder, self.human_decoder, self.keypoints_decoder,
        ]
        
        # Freeze all encoders and decoders
        for module in encoders_decoders:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        
        # Keep action decoder trainable
        for param in self.action_decoder.parameters():
            param.requires_grad = True
        self.action_decoder.train()
        
        print(f"[INFO] Actor_Triple_AE: Frozen all encoders and decoders")
        print(f"       - Frozen: robot_encoder, human_encoder, keypoints_encoder")
        print(f"       -         robot_decoder, human_decoder, keypoints_decoder")
        print(f"       - Trainable: action_decoder, proprioceptive_normalizer")
    
    
    def freeze_for_finetune(self, finetune_networks: list[str] = ['human_encoder', 'human_decoder']):
        """
        灵活地冻结指定组件：仅解冻且开放列表中被选中的网络进行训练，其余全部锁死梯度。
        
        参数:
            finetune_networks: 需要保持开放训练的网络模块名称列表。
                               选项包含：'robot_encoder', 'human_encoder', 'keypoints_encoder',
                                        'robot_decoder', 'human_decoder', 'keypoints_decoder',
                                        'action_decoder', 'proprioceptive_normalizer'
        """
        self._frozen_for_finetune = True
        
        all_networks = {
            'robot_encoder': self.robot_encoder,
            'human_encoder': self.human_encoder,
            'keypoints_encoder': self.keypoints_encoder,
            'robot_decoder': self.robot_decoder,
            'human_decoder': self.human_decoder,
            'keypoints_decoder': self.keypoints_decoder,
            'action_decoder': self.action_decoder,
            'proprioceptive_normalizer': self.proprioceptive_normalizer,
        }
        
        invalid_networks = set(finetune_networks) - set(all_networks.keys())
        if invalid_networks:
            raise ValueError(f"Invalid network names: {invalid_networks}")
        
        for network_name, network_module in all_networks.items():
            for param in network_module.parameters():
                param.requires_grad = False
            network_module.eval()
        
        for network_name in finetune_networks:
            for param in all_networks[network_name].parameters():
                param.requires_grad = True
            all_networks[network_name].train()
        
        print(f"[INFO] Actor_Triple_AE: Frozen networks for finetuning")
        print(f"       - Trainable: {finetune_networks}")
        print(f"       - Frozen: {[n for n in all_networks.keys() if n not in finetune_networks]}")
    
    def train(self, mode: bool = True):
        """Override train() to manage frozen modules during finetuning."""
        super().train(mode)


class Critic_Triple_AE(nn.Module):
    """
    负责价值(Value)预估和评判的 Critic 网络，配合特制的 Triple_AE 模式使用。
    
    采用标准的多层感知机(MLP)结构, 但其观察数据并非原始数据, 而是接收来自【Triple_AE 编码器】输出的
    “隐变量”(Latent Vector) 加上 “本体感知”(Proprioceptive State) 后进行全连接并预测环境势能。
    
    网络结构:
        输入层: 隐变量向量尺度(latent_dim) + 本体维度(critic_sp_dim)
        隐藏层: critic_hidden_dims
        输出层: 环境状态评分预估 (数值标量)
    """
    
    def __init__(
        self,
        latent_dim: int,
        critic_sp_dim: int,
        critic_hidden_dims: list[int] = None,
        activation: str = "elu",
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.critic_sp_dim = critic_sp_dim
        
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 256]
        
        activation_fn = resolve_nn_activation(activation)
        
        # Proprioceptive State Normalizer
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(critic_sp_dim,))
        
        # Critic network: [latent + proprioceptive_state] -> value
        critic_input_dim = latent_dim + critic_sp_dim
        critic_layers = []
        
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i+1]))
            critic_layers.append(activation_fn)
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        
        self.critic_mlp = nn.Sequential(*critic_layers)
    
    def forward(self, z, x_sp):
        """Forward pass of the critic.
        
        Args:
            z: Latent vector from Triple_AE encoder of shape (batch_size, latent_dim)
            x_sp: Proprioceptive state of shape (batch_size, critic_sp_dim)
        
        Returns:
            value: Value estimate of shape (batch_size, 1)
        """
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        return self.critic_mlp(torch.cat([z, x_sp_normalized], dim=-1))


class ActorCritic_Triple_AE(nn.Module):
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # actor triple_ae specific
        actor_sg_dim: int = None,
        actor_sh_dim: int = 0,
        actor_sk_dim: int = 0,
        latent_dim: int = 32,
        robot_encoder_hidden_dims: list[int] = None,
        human_encoder_hidden_dims: list[int] = None,
        keypoints_encoder_hidden_dims: list[int] = None,
        robot_decoder_hidden_dims: list[int] = None,
        human_decoder_hidden_dims: list[int] = None,
        keypoints_decoder_hidden_dims: list[int] = None,
        activate_signals: Literal["robot", "smplx", "keypoints"] = "robot",
        **kwargs,
    ):
        if kwargs:
            print(f"[WARNING] Unexpected kwargs: {kwargs}")
        
        super().__init__()
        activation_fn = resolve_nn_activation(activation)
        
        # Set default hidden dims if not provided
        if robot_encoder_hidden_dims is None:
            robot_encoder_hidden_dims = [512, 256]
        if human_encoder_hidden_dims is None:
            human_encoder_hidden_dims = [512, 256]
        if keypoints_encoder_hidden_dims is None:
            keypoints_encoder_hidden_dims = [512, 256]
        if robot_decoder_hidden_dims is None:
            robot_decoder_hidden_dims = [256, 512]
        if human_decoder_hidden_dims is None:
            human_decoder_hidden_dims = [256, 512]
        if keypoints_decoder_hidden_dims is None:
            keypoints_decoder_hidden_dims = [256, 512]
        
        self.activate_signals = activate_signals
        self.actor_sg_dim = actor_sg_dim
        self.actor_sh_dim = actor_sh_dim
        self.actor_sk_dim = actor_sk_dim
        
        # ==================== Actor: Triple_AE Policy ====================
        self.actor = Actor_Triple_AE(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_sg_dim=actor_sg_dim,
            actor_sh_dim=actor_sh_dim,
            actor_sk_dim=actor_sk_dim,
            latent_dim=latent_dim,
            robot_encoder_hidden_dims=robot_encoder_hidden_dims,
            human_encoder_hidden_dims=human_encoder_hidden_dims,
            keypoints_encoder_hidden_dims=keypoints_encoder_hidden_dims,
            robot_decoder_hidden_dims=robot_decoder_hidden_dims,
            human_decoder_hidden_dims=human_decoder_hidden_dims,
            keypoints_decoder_hidden_dims=keypoints_decoder_hidden_dims,
            action_decoder_hidden_dims=actor_hidden_dims,
            activation=activation,
            activate_signals=activate_signals,
        )
        
        # ==================== Critic: Value function ====================
        critic_sp_dim = num_critic_obs - actor_sg_dim - actor_sh_dim - actor_sk_dim
        
        self.critic = Critic_Triple_AE(
            latent_dim=latent_dim,
            critic_sp_dim=critic_sp_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        
        # ==================== Action noise ====================
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]
        
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update the distribution based on observations."""
        mean = self.actor(observations)
        
        if self.noise_std_type == "scalar":
            std = torch.exp(self.log_std)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")
        
        self.distribution = Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value from critic using activate_signals to choose modality."""
        x_sp_critic = critic_observations[:, self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim:]
        
        if self.activate_signals == "robot":
            x_sg_critic = critic_observations[:, :self.actor_sg_dim]
            z = self.actor.encode_robot(x_sg_critic)
        elif self.activate_signals == "smplx":
            x_sh_critic = critic_observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
            z = self.actor.encode_smplx(x_sh_critic)
        elif self.activate_signals == "keypoints":
            x_sk_critic = critic_observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
            z = self.actor.encode_keypoints(x_sk_critic)
        else:
            raise ValueError(f"Invalid activate_signals: {self.activate_signals}")
        
        # Detach latent to prevent critic gradients from flowing back to the encoder
        # The encoder is updated through auxiliary losses (reconstruction, alignment, consistency)
        z = z.detach()
        
        value = self.critic(z, x_sp_critic)
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model."""
        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def get_alignment_loss(self, observations):
        """
        计算提取自三方观察值之间的联合在隐层特征空间的【对齐损失 (Alignment Loss)】。
        
        对齐损失公式 = w1 * MSE(z_robot, z_human) + w2 * MSE(z_robot, z_keypoints) + w3 * MSE(z_human, z_keypoints)
        
        参数:
            observations: 完整的复合观测空间(包含三者加本体系)
        
        返回:
            包含各项独立对齐误差及加权总和数值的字典：
            {
                'alignment_sg_sh': 机器人与人类中间态隐层特征的 MSE 对齐值,
                'alignment_sg_sk': 机器人与关键点中间态隐层特征的 MSE 对齐值,
                'alignment_sh_sk': 人类与关键点中间态隐层特征的 MSE 对齐值,
                'alignment_total': 将上述内容用预设权重进行求和出的总对齐残差参数
            }
        """
        # Encode all three modalities
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        # Compute three-way alignment losses
        align_sg_sh = torch.mean((z_sg - z_sh) ** 2)
        align_sg_sk = torch.mean((z_sg - z_sk) ** 2)
        align_sh_sk = torch.mean((z_sh - z_sk) ** 2)
        
        # Weighted combination (you can adjust weights)
        alignment_total = align_sg_sh + 0.8 * align_sg_sk + 0.6 * align_sh_sk
        
        return {
            'alignment_sg_sh': align_sg_sh,
            'alignment_sg_sk': align_sg_sk,
            'alignment_sh_sk': align_sh_sk,
            'alignment_total': alignment_total
        }
    
    def get_reconstruction_loss(self, observations):
        """Compute reconstruction losses for all three decoders.
        
        Args:
            observations: Input observations of shape (batch_size, num_critic_obs)
        
        Returns:
            Dictionary containing:
            {
                'recon_sg': MSE reconstruction for robot state,
                'recon_sh': MSE reconstruction for human state,
                'recon_sk': MSE reconstruction for keypoints state,
                'recon_total': Sum of all reconstruction losses
            }
        """
        # Extract state components
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sk = observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        
        # Encode all three modalities
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        # Reconstruct and compute losses
        x_sg_recon = self.actor.decode_robot(z_sg)
        recon_sg = torch.mean((x_sg_recon - x_sg) ** 2)
        
        x_sh_recon = self.actor.decode_human(z_sh)
        recon_sh = torch.mean((x_sh_recon - x_sh) ** 2)
        
        x_sk_recon = self.actor.decode_keypoints(z_sk)
        recon_sk = torch.mean((x_sk_recon - x_sk) ** 2)
        
        recon_total = recon_sg + recon_sh + recon_sk
        
        return {
            'recon_sg': recon_sg,
            'recon_sh': recon_sh,
            'recon_sk': recon_sk,
            'recon_total': recon_total
        }
    
    def get_consistency_loss(self, observations):
        """
        专门计算用于确保 Triple_AE 内置映射转换完美平滑的【跨模态一致性损失 (Consistency Loss)】。
        
        此损失函数本质是确保当把信息放入一个模态(Modality)编码，强行用"另一个"模态对应的解码器
        解码还原后，在物理意义和特征信息上它们依旧能保持紧致对流而不发生“幻觉”偏离。
        
        核心操作逻辑（取决于当前激活驱动目标 activate_signals）:
        若激活目标为 "robot"(机器人):
            - 编码人类数据和关键点特征
            - 将它们投入“机器人解码器(robot_decoder)”解码
            - 一致性约束结果 = MSE(decoder_robot(z_human), x_机器人状态) + MSE(...)
            
        若为 SMPLX (人类视觉网格) / keypoints 等，类似交叉反推操作。
        """
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sk = observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        mse_loss = torch.nn.MSELoss()
        
        if self.activate_signals == "robot":
            # Decode human and keypoints latents with robot decoder
            x_sg_from_human = self.actor.decode_robot(z_sh)
            x_sg_from_keypoints = self.actor.decode_robot(z_sk)
            
            consist_1 = mse_loss(x_sg_from_human, x_sg)
            consist_2 = mse_loss(x_sg_from_keypoints, x_sg)
            consist_total = consist_1 + consist_2
            
        elif self.activate_signals == "smplx":
            # Decode robot and keypoints latents with human decoder
            x_sh_from_robot = self.actor.decode_human(z_sg)
            x_sh_from_keypoints = self.actor.decode_human(z_sk)
            
            consist_1 = mse_loss(x_sh_from_robot, x_sh)
            consist_2 = mse_loss(x_sh_from_keypoints, x_sh)
            consist_total = consist_1 + consist_2
            
        elif self.activate_signals == "keypoints":
            # Decode robot and human latents with keypoints decoder
            x_sk_from_robot = self.actor.decode_keypoints(z_sg)
            x_sk_from_human = self.actor.decode_keypoints(z_sh)
            
            consist_1 = mse_loss(x_sk_from_robot, x_sk)
            consist_2 = mse_loss(x_sk_from_human, x_sk)
            consist_total = consist_1 + consist_2
            
        else:
            raise ValueError(f"Unknown activate_signals: {self.activate_signals}")
        
        return {
            'consistency_loss1': consist_1,
            'consistency_loss2': consist_2,
            'consistency_total': consist_total
        }
    
    def get_auxiliary_loss(self, observations, loss_items: list[str] = ['alignment', 'reconstruction', 'consistency']):
        """
        高效率在一次前向推导环节中打包计算并整合出所有的【Triple_AE 特有辅助损失】。
        
        这规避了多次重复计算编码器的性能浪费，直接针对 `[对齐, 重建, 一致性]` 输出梯度追踪值。
        
        返回:
            记录各种计算好损失值的字典(Dictionary), 用以直接挂轨累加回在外部计算中的 PPO 主干 Loss 体系上。
        """
        losses = {
            "alignment": None,
            "reconstruction_sg": None,
            "reconstruction_sh": None,
            "reconstruction_sk": None,
            "consistency": None
        }
        
        # Extract states once
        x_sg = observations[:, :self.actor_sg_dim]
        x_sh = observations[:, self.actor_sg_dim:self.actor_sg_dim + self.actor_sh_dim]
        x_sk = observations[:, self.actor_sg_dim + self.actor_sh_dim:self.actor_sg_dim + self.actor_sh_dim + self.actor_sk_dim]
        
        # Encode all modalities once
        z_sg = self.actor.encode_robot(observations)
        z_sh = self.actor.encode_smplx(observations)
        z_sk = self.actor.encode_keypoints(observations)
        
        mse_loss = torch.nn.MSELoss()
        
        # Compute reconstruction losses if requested
        if 'reconstruction' in loss_items:
            x_sg_recon = self.actor.decode_robot(z_sg)
            recon_sg = torch.mean((x_sg_recon - x_sg) ** 2)
            losses['reconstruction_sg'] = recon_sg
            
            x_sh_recon = self.actor.decode_human(z_sh)
            recon_sh = torch.mean((x_sh_recon - x_sh) ** 2)
            losses['reconstruction_sh'] = recon_sh
            
            x_sk_recon = self.actor.decode_keypoints(z_sk)
            recon_sk = torch.mean((x_sk_recon - x_sk) ** 2)
            losses['reconstruction_sk'] = recon_sk
        
        # Compute alignment loss if requested
        if 'alignment' in loss_items:
            align_sg_sh = torch.mean((z_sg - z_sh) ** 2)
            align_sg_sk = torch.mean((z_sg - z_sk) ** 2)
            align_sh_sk = torch.mean((z_sh - z_sk) ** 2)
            
            # Weighted three-way alignment (using actor's alignment weights)
            alignment_loss = align_sg_sh + align_sg_sk + align_sh_sk
            losses['alignment'] = alignment_loss
        
        # Compute consistency loss if requested
        if 'consistency' in loss_items:
            if self.activate_signals == "robot":
                x_sg_from_human = self.actor.decode_robot(z_sh)
                x_sg_from_keypoints = self.actor.decode_robot(z_sk)
                consistency_loss = (
                    mse_loss(x_sg_from_human, x_sg) +
                    mse_loss(x_sg_from_keypoints, x_sg)
                ) / 2.0
            elif self.activate_signals == "smplx":
                x_sh_from_robot = self.actor.decode_human(z_sg)
                x_sh_from_keypoints = self.actor.decode_human(z_sk)
                consistency_loss = (
                    mse_loss(x_sh_from_robot, x_sh) +
                    mse_loss(x_sh_from_keypoints, x_sh)
                ) / 2.0
            elif self.activate_signals == "keypoints":
                x_sk_from_robot = self.actor.decode_keypoints(z_sg)
                x_sk_from_human = self.actor.decode_keypoints(z_sh)
                consistency_loss = (
                    mse_loss(x_sk_from_robot, x_sk) +
                    mse_loss(x_sk_from_human, x_sk)
                ) / 2.0
            else:
                raise ValueError(f"Unknown activate_signals: {self.activate_signals}")
            
            losses['consistency'] = consistency_loss
        
        return losses

###########################################
# ActorCritic_Triple_AE_Single_Finetune 
#   A variant of ActorCritic_Triple_AE for single modality finetuning.
#   usage: `freeze` parameter to freeze cmd encoders/decoders only. finetune the action decoder. 
###########################################

class Actor_Triple_AE_Single_Finetune(nn.Module):
    """
    用于单模态微调(Finetuning)的Actor网络, 支持冻结编码器/解码器。
    
    该Actor网络使用单一的指令编码器(已预训练)和动作解码器。
    在微调期间，指令(cmd)编码器可以被冻结，此时只对动作解码器进行训练。
    
    网络结构:
        cmd_state (指令状态) -> cmd_encoder -> latent_dim (隐空间维度)
        latent_dim + proprioceptive_state -> action_decoder -> actions (策略动作)
        latent_dim -> cmd_decoder -> cmd_state_recon (用于补充可选的状态重构损失)
    """
    
    def __init__(
        self,
        num_actor_obs: int,
        num_actions: int,
        actor_cmd_dim: int,
        latent_dim: int = 32,
        cmd_encoder_hidden_dims: list[int] = None,
        cmd_decoder_hidden_dims: list[int] = None,
        action_decoder_hidden_dims: list[int] = None,
        activation: str = "elu",
        freeze: bool = True,
    ):
        super().__init__()
        
        self.num_actor_obs = num_actor_obs
        self.actor_cmd_dim = actor_cmd_dim
        self.num_actions = num_actions
        self.latent_dim = latent_dim
        self.freeze = freeze
        
        # Set default hidden dimensions if not provided
        if cmd_encoder_hidden_dims is None:
            cmd_encoder_hidden_dims = [512, 256]
        if cmd_decoder_hidden_dims is None:
            cmd_decoder_hidden_dims = [256, 512]
        if action_decoder_hidden_dims is None:
            action_decoder_hidden_dims = [256, 256, 256]
        
        activation_fn = resolve_nn_activation(activation)
        
        # ==================== Cmd Encoder ====================
        # Input: cmd_state (actor_cmd_dim) -> Output: latent_dim
        cmd_encoder_layers = []
        cmd_encoder_layers.append(nn.Linear(actor_cmd_dim, cmd_encoder_hidden_dims[0]))
        cmd_encoder_layers.append(activation_fn)
        for i in range(len(cmd_encoder_hidden_dims) - 1):
            cmd_encoder_layers.append(nn.Linear(cmd_encoder_hidden_dims[i], cmd_encoder_hidden_dims[i+1]))
            cmd_encoder_layers.append(activation_fn)
        cmd_encoder_layers.append(nn.Linear(cmd_encoder_hidden_dims[-1], latent_dim))
        self.cmd_encoder = nn.Sequential(*cmd_encoder_layers)
        
        # ==================== Cmd Decoder ====================
        # Input: latent_dim -> Output: cmd_state (actor_cmd_dim)
        cmd_decoder_layers = []
        cmd_decoder_layers.append(nn.Linear(latent_dim, cmd_decoder_hidden_dims[0]))
        cmd_decoder_layers.append(activation_fn)
        for i in range(len(cmd_decoder_hidden_dims) - 1):
            cmd_decoder_layers.append(nn.Linear(cmd_decoder_hidden_dims[i], cmd_decoder_hidden_dims[i+1]))
            cmd_decoder_layers.append(activation_fn)
        cmd_decoder_layers.append(nn.Linear(cmd_decoder_hidden_dims[-1], actor_cmd_dim))
        self.cmd_decoder = nn.Sequential(*cmd_decoder_layers)
        
        # ==================== Proprioceptive State Normalizer ====================
        actor_sp_dim = num_actor_obs - actor_cmd_dim
        self.proprioceptive_normalizer = EmpiricalNormalization(shape=(actor_sp_dim,))
        
        # ==================== Action Decoder ====================
        # Input: latent_dim + actor_sp_dim -> Output: num_actions
        action_layers = []
        action_layers.append(nn.Linear(latent_dim + actor_sp_dim, action_decoder_hidden_dims[0]))
        action_layers.append(activation_fn)
        for i in range(len(action_decoder_hidden_dims) - 1):
            action_layers.append(nn.Linear(action_decoder_hidden_dims[i], action_decoder_hidden_dims[i+1]))
            action_layers.append(activation_fn)
        action_layers.append(nn.Linear(action_decoder_hidden_dims[-1], num_actions))
        self.action_decoder = nn.Sequential(*action_layers)
        
        # Apply freeze if specified
        if freeze:
            self._apply_freeze()
    
    def _apply_freeze(self):
        """冻结指令(cmd)的编码器和解码器梯度的流动，并保持动作(action)解码器可训练。"""
        # 冻结(Freeze)指令编码器
        for param in self.cmd_encoder.parameters():
            param.requires_grad = False
        self.cmd_encoder.eval()
        
        # 冻结(Freeze)指令状态解码器
        for param in self.cmd_decoder.parameters():
            param.requires_grad = False
        self.cmd_decoder.eval()
        
        print(f"[INFO] Actor_Triple_AE_Single_Finetune: Frozen cmd_encoder and cmd_decoder")
        print(f"       - Trainable: action_decoder, proprioceptive_normalizer")
    
    def forward(self, x):
        """通过单模态 Actor 的主要前向计算。
        
        参数:
            x: 输入特征张量，形状为 (batch_size, num_actor_obs)
               其结构组成应该为: [cmd_state | proprioceptive_state]
        
        返回:
            actions: 预测推导的动作，形状为 (batch_size, num_actions)
        """
        # 分离组合好的输入层: [cmd_state(指令空间) | proprioceptive_state(本体观测空间)]
        x_cmd = x[:, :self.actor_cmd_dim]
        x_sp = x[:, self.actor_cmd_dim:]
        
        # 将指令向量编码到隐层特征
        z = self.cmd_encoder(x_cmd)
        # 取消隐层特征梯度的回传(detach)以避免训练策略Actor损失时PPO影响到预训练良好的Encoder
        z_for_action = z.detach()
        
        # 标准化本体传感状态
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        
        # 向量拼接后经过解码器输出具体的执行 Action
        actions = self.action_decoder(torch.cat([z_for_action, x_sp_normalized], dim=-1))
        
        return actions
    
    def encode(self, x):
        """把指令状态(cmd_state)编码投影成隐变量(latent)。
        
        参数:
            x: 输入张量，形状可以为 (batch_size, num_actor_obs) 或者直接是 (batch_size, actor_cmd_dim)
        
        返回:
            z: 特征隐向量，形状为 (batch_size, latent_dim)
        """
        if x.shape[-1] == self.num_actor_obs:
            x_cmd = x[:, :self.actor_cmd_dim]
        else:
            x_cmd = x
        
        return self.cmd_encoder(x_cmd)
    
    def decode_cmd(self, z):
        """解码隐空间向量进而针对性还原重构出指令(cmd)状态特征。
        
        参数:
            z: 隐层变量张量，维数形状为 (batch_size, latent_dim)
        
        返回:
            cmd_recon: 被重新构建刻画出的指令状态对象，其形状为 (batch_size, actor_cmd_dim)
        """
        return self.cmd_decoder(z)
    
    def decode_action(self, z, x_sp):
        """根据隐层特征及本体感知状态解码生成动作。
        
        参数:
            z: 隐层向量，形状为 (batch_size, latent_dim)
            x_sp: 本体感知状态，形状为 (batch_size, actor_sp_dim)
        
        返回:
            actions: 网络的动作预测输出，形状为 (batch_size, num_actions)
        """
        x_sp_normalized = self.proprioceptive_normalizer(x_sp)
        # 调用具有 detach (无梯度游移影响的) 隐变量来保证动作解码推导演算的梯度不会向上游污染编码器(Encoder)
        return self.action_decoder(torch.cat([z.detach(), x_sp_normalized], dim=-1))
    
    def forward_exporter(self, cmd_state, proprioceptive_state):
        """针对底层部署环境执行脱钩输入的专属微调前向传播方法。
        
        参数:
            cmd_state: 张量对象形式代表的物理域命令(cmd)，形状为 (batch_size, actor_cmd_dim)
            proprioceptive_state: 此轮推演下的本体观测环境感知度信息张量
        
        返回:
            actions: 控制模块反馈的具体行动向量 (batch_size, num_actions)
        """
        z = self.cmd_encoder(cmd_state)
        z_for_action = z.detach()
        proprioceptive_state_normalized = self.proprioceptive_normalizer(proprioceptive_state)
        actions = self.action_decoder(torch.cat([z_for_action, proprioceptive_state_normalized], dim=-1))
        return actions
    
    def train(self, mode: bool = True):
        """通过重写框架 train()，即使网络整体设定为进入训练，冻结(frozen)状态下的组块依然保在 eval() 无状态缓存模式下。"""
        super().train(mode)
        if self.freeze:
            self.cmd_encoder.eval()
            self.cmd_decoder.eval()
        return self


class ActorCritic_Triple_AE_Single_Finetune(nn.Module):
    """
    为了单模态应用精细微调(finetuning)而设立的结构变体版 ActorCritic 评估策略组系。
    
    此变种形态在创建时就默认将指令层(cmd)端到端特征重构系统(编码器/解码器)冻结冷启动，
    模型仅需学习如何依靠当前确定的这套隐层空间推断行动逻辑。这对于把预训练完好的大结构多模态动作迁移匹配
    至固定受限的单种信号设备中是高性价比的。
    
    其中的 Critic 评价网络共享使用完全相同的且来源冻结编码器转换后的隐藏数据维。
    """
    is_recurrent = False
    
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # actor triple_ae specific
        actor_cmd_dim: int = None,
        latent_dim: int = 32,
        cmd_encoder_hidden_dims: list[int] = None,
        cmd_decoder_hidden_dims: list[int] = None,
        freeze: bool = True,
        encoder_key: Literal["robot", "human", "keypoints"] = "robot",
        **kwargs,
    ):
        if kwargs:
            print(f"[WARNING] Unexpected kwargs: {kwargs}")
        
        super().__init__()
        
        self.actor_cmd_dim = actor_cmd_dim
        self.latent_dim = latent_dim
        self.freeze = freeze
        self.encoder_key = encoder_key
        
        # Set default hidden dims if not provided
        if cmd_encoder_hidden_dims is None:
            cmd_encoder_hidden_dims = [512, 256]
        if cmd_decoder_hidden_dims is None:
            cmd_decoder_hidden_dims = [256, 512]
        
        # ==================== Actor: Single Modality AE Policy ====================
        self.actor = Actor_Triple_AE_Single_Finetune(
            num_actor_obs=num_actor_obs,
            num_actions=num_actions,
            actor_cmd_dim=actor_cmd_dim,
            latent_dim=latent_dim,
            cmd_encoder_hidden_dims=cmd_encoder_hidden_dims,
            cmd_decoder_hidden_dims=cmd_decoder_hidden_dims,
            action_decoder_hidden_dims=actor_hidden_dims,
            activation=activation,
            freeze=freeze,
        )
        
        # ==================== Critic: Value function ====================
        critic_sp_dim = num_critic_obs - actor_cmd_dim
        
        self.critic = Critic_Triple_AE(
            latent_dim=latent_dim,
            critic_sp_dim=critic_sp_dim,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
        )
        
        print(f"Actor: {self.actor}")
        print(f"Critic: {self.critic}")
        
        # ==================== Action noise ====================
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.ones(num_actions) * torch.log(torch.tensor(init_noise_std)))
        else:
            raise ValueError(f"Invalid noise_std_type: {noise_std_type}")
        
        # Action distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
    
    def reset(self, dones=None):
        pass
    
    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    def update_distribution(self, observations):
        """Update the distribution based on observations."""
        mean = self.actor(observations)
        
        if self.noise_std_type == "scalar":
            std = torch.exp(self.log_std)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std)
        else:
            raise ValueError(f"Invalid noise_std_type: {self.noise_std_type}")
        
        self.distribution = Normal(mean, std)
    
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean
    
    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value from critic."""
        x_cmd = critic_observations[:, :self.actor_cmd_dim]
        x_sp_critic = critic_observations[:, self.actor_cmd_dim:]
        
        z = self.actor.encode(x_cmd)
        # Detach latent to prevent critic gradients from flowing back to the encoder
        # When encoder is frozen, this has no effect; when unfrozen, it separates gradient paths
        z = z.detach()
        value = self.critic(z, x_sp_critic)
        return value
    
    def load_state_dict(self, state_dict, strict=True):
        """Load the parameters of the actor-critic model."""
        super().load_state_dict(state_dict, strict=strict)
        return True
    
    def get_reconstruction_loss(self, observations):
        """Compute reconstruction loss for cmd state.
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
        
        Returns:
            Dictionary containing:
            {
                'recon_cmd': MSE reconstruction for cmd state,
            }
        """
        x_cmd = observations[:, :self.actor_cmd_dim]
        
        z = self.actor.encode(observations)
        x_cmd_recon = self.actor.decode_cmd(z)
        
        recon_cmd = torch.mean((x_cmd_recon - x_cmd) ** 2)
        
        return {
            'recon_cmd': recon_cmd,
        }
    
    def get_auxiliary_loss(self, observations, loss_items: list[str] = ['reconstruction']):
        """Compute auxiliary losses for single modality finetuning.
        
        Args:
            observations: Input observations of shape (batch_size, num_actor_obs)
            loss_items: List of loss types to compute.
                       Options: 'reconstruction'
        
        Returns:
            Dictionary with computed losses.
        """
        losses = {
            "reconstruction_cmd": None,
        }
        
        if 'reconstruction' in loss_items:
            x_cmd = observations[:, :self.actor_cmd_dim]
            z = self.actor.encode(observations)
            x_cmd_recon = self.actor.decode_cmd(z)
            losses['reconstruction_cmd'] = torch.mean((x_cmd_recon - x_cmd) ** 2)
        
        return losses
    
    def load_state_dict(self, state_dict, strict=True):
        """Load pretrained weights from ActorCritic_Triple_AE model.
        
        Maps the selected encoder/decoder to cmd_encoder/cmd_decoder and
        removes unused encoder/decoder weights.
        
        Args:
            state_dict: State dict from a pre-trained ActorCritic_Triple_AE model.
            strict: Whether to strictly enforce that the keys match.
        """
        encoder_key = f"{self.encoder_key}_encoder"
        decoder_key = f"{self.encoder_key}_decoder"
        
        # All encoder/decoder keys in Triple_AE
        all_encoder_keys = ["robot_encoder", "human_encoder", "keypoints_encoder"]
        all_decoder_keys = ["robot_decoder", "human_decoder", "keypoints_decoder"]
        
        # Keys to delete (encoders/decoders we don't need)
        delete_encoder_keys = [k for k in all_encoder_keys if k != encoder_key]
        delete_decoder_keys = [k for k in all_decoder_keys if k != decoder_key]
        
        # Build new state dict with remapped keys
        # Patterns for keys to delete
        delete_patterns = [re.compile(rf'actor\.{k}\.') for k in delete_encoder_keys + delete_decoder_keys]
        
        # Patterns for keys to remap
        encoder_pattern = re.compile(rf'(actor\.){encoder_key}(\..*)')
        decoder_pattern = re.compile(rf'(actor\.){decoder_key}(\..*)')
        
        new_state_dict = {}
        for key, value in state_dict.items():
            # Skip keys from unused encoders/decoders
            if any(p.search(key) for p in delete_patterns):
                continue
            
            # Remap selected encoder to cmd_encoder
            match = encoder_pattern.match(key)
            if match:
                new_key = f"{match.group(1)}cmd_encoder{match.group(2)}"
                new_state_dict[new_key] = value
                continue
            
            # Remap selected decoder to cmd_decoder
            match = decoder_pattern.match(key)
            if match:
                new_key = f"{match.group(1)}cmd_decoder{match.group(2)}"
                new_state_dict[new_key] = value
                continue
            
            # Keep other keys as-is
            new_state_dict[key] = value
        
        # Load the remapped state dict
        super().load_state_dict(new_state_dict, strict=strict)
        
        print(f"[INFO] Loaded pretrained weights with encoder_key='{self.encoder_key}'")
        print(f"       - Mapped: actor.{encoder_key} -> actor.cmd_encoder")
        print(f"       - Mapped: actor.{decoder_key} -> actor.cmd_decoder")
        print(f"       - Deleted: {delete_encoder_keys + delete_decoder_keys}")
        
        # Re-apply freeze after loading
        if self.freeze:
            self.actor._apply_freeze()
        
        return True
        
        