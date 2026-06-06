# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Literal

import onnx
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl.exporter import _OnnxPolicyExporter

from wbc_robot.tasks.manager_based.wbc_robot.mdp import MotionCommand


# ---------------------------------------------------------
# 统一的策略导出函数 (Main Export Function)
# 作用：根据传入的 task_type 参数，选择相应的具体 Exporter 类
# (如从单个动作、多动作到GAEMimic多模态追踪)，
# 将训练好的 PyTorch 神经网络模型 (Actor-Critic) 转化为跨平台的 ONNX 格式。
# ONNX 是一种开放模型格式，它可以在仿真系统外、甚至是真实机器人硬件中进行快速推理。
# ---------------------------------------------------------
def export_motion_policy_as_onnx(
    env: ManagerBasedRLEnv | None,
    actor_critic: object,
    path: str,
    task_type: Literal["single_motion", "multi_motion", "gae_mimic"] = "multi_motion",
    gaemimic_task: Literal["robot", "human", "keypoints"] = "robot",
    normalizer: object | None = None,
    filename="policy.onnx",
    verbose=False,
):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if task_type == "single_motion":
        policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
    elif task_type == "multi_motion":
        policy_exporter = _Onnx_MultiMotion_PolicyExporter(env, actor_critic, normalizer, verbose)
    elif task_type == "gae_mimic":
        policy_exporter = _Onnx_GAEMimic_PolicyExporter(actor_critic, env, normalizer, verbose, task=gaemimic_task)
    else:
        raise ValueError(f"Unknown policy export type: {task_type}")

    policy_exporter.export(path, filename)


# ---------------------------------------------------------
# 1. 单个动作的 ONNX 导出器 (Single Motion Policy Exporter) [早期简单实现]
# 作用：除了将被训练的 Actor（策略网络）导出外，
# 它还会「直接内嵌」那一个动作数据集 (关节位置、刚体轨迹等) 到模型之中。
# 这样在模型推理时，只要传入目标 `time_step` (时间步)，
# 推理模型除了给出动作外，还会返回那一个时刻动捕数据对应的参考位姿。
# 注意：这种把数据硬编码进神经网络的做法在多动作时会大大膨胀模型体积。
# ---------------------------------------------------------
class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

    def forward(self, x, time_step):
        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(x)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.actor[0].in_features)
        time_step = torch.zeros(1, 1)
        torch.onnx.export(
            self,
            (obs, time_step),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs", "time_step"],
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


# ---------------------------------------------------------
# 2. 单个动作的 ONNX 导出器 (改进版: 支持分开的观察空间输入)
# 作用：与上面的类逻辑相似并将参考动作内嵌在模型里，
# 改进在于它支持 env.observation_manager 分离的多块观察状态 (Observations)。
# 它通过 `*args` 动态收集不同的观察数据拼成完整的 obs 张量并喂给 Actor。
# ---------------------------------------------------------
class _Onnx_Motion_PolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)
        cmd: MotionCommand = env.command_manager.get_term("motion")

        self.joint_pos = cmd.motion.joint_pos.to("cpu")
        self.joint_vel = cmd.motion.joint_vel.to("cpu")
        self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
        self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
        self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
        self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")
        self.time_step_total = self.joint_pos.shape[0]

        self.observation_names = env.observation_manager.active_terms["policy"]
        group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
        self.observation_dims = [dims[-1] for dims in group_obs_term_dim]

    def forward(self, *args):
        # args contains separate observation terms
        obs = torch.cat(args[:-1], dim=-1)
        time_step = args[-1]

        time_step_clamped = torch.clamp(time_step.long().squeeze(-1), max=self.time_step_total - 1)
        return (
            self.actor(self.normalizer(obs)),
            self.joint_pos[time_step_clamped],
            self.joint_vel[time_step_clamped],
            self.body_pos_w[time_step_clamped],
            self.body_quat_w[time_step_clamped],
            self.body_lin_vel_w[time_step_clamped],
            self.body_ang_vel_w[time_step_clamped],
        )

    def export(self, path, filename):
        self.to("cpu")

        # Create separate dummy inputs for each observation term
        dummy_inputs = []
        for dim in self.observation_dims:
            dummy_inputs.append(torch.zeros(1, dim))

        # Add time_step as the last input
        time_step = torch.zeros(1, 1)
        dummy_inputs.append(time_step)

        input_names = list(self.observation_names) + ["time_step"]

        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=input_names,
            output_names=[
                "actions",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ],
            dynamic_axes={},
        )


# ---------------------------------------------------------
# 3. 多动作的 ONNX 导出器 (Multi-Motion Policy Exporter) [最常用的核心实现]
# 作用：这专为包含成百上千个动捕文件的多动作追踪而设计 (我们代码中主用的版本)。
# 因为动作太多了，不可能也不应该把 GB 级别的动作数据一股脑打包进 ONNX。
# 
# 所以在这个导出器里，它干脆 剥离了时间步参数和内部动作查询，
# 只保留了纯粹的 Actor 模型：
# 输入：当前系统的各项观察量拼接； 
# 输出：机器人的动作指令(PID驱动/力矩等_具体看配置)。
# 
# 它的体积极小，运行极快，在实机部署中，动捕位姿的传入是依靠外围代码提供的，
# 而不是在这个 `.onnx` 内部自己查表的。
# ---------------------------------------------------------
class _Onnx_MultiMotion_PolicyExporter(_OnnxPolicyExporter):
    def __init__(self, env: ManagerBasedRLEnv, actor_critic, normalizer=None, verbose=False):
        super().__init__(actor_critic, normalizer, verbose)

        self.observation_names = env.observation_manager.active_terms["policy"]
        group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
        self.observation_dims = [dims[-1] for dims in group_obs_term_dim]

        if verbose:
            print(f"Observation names: {self.observation_names}")
            print(f"Observation dims: {self.observation_dims}")

    def forward(self, *args):
        obs = torch.cat(args, dim=-1)
        return self.actor(self.normalizer(obs))

    def export(self, path, filename):
        self.to("cpu")

        # Create separate dummy inputs for each observation term
        dummy_inputs = []
        for dim in self.observation_dims:
            dummy_inputs.append(torch.zeros(1, dim))

        input_names = list(self.observation_names)

        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=input_names,
            output_names=["actions"],
            dynamic_axes={},
        )


def register_forward(version):
    """
    装饰器：用于标记此方法属于哪个特定任务（如 "robot", "human" 等）。
    它会在函数上打个 `_forward_version` 的标签，以便后续给类实例化时匹配提取。
    """
    def decorator(func):
        func._forward_version = version
        return func

    return decorator


def select_forward(cls):
    """
    类装饰器：在类实例化时，根据用户传入的 `task` 属性，动态挑选与之匹配的前向逻辑方法，强行将其绑定为 `self.forward`。
    这样做的目的是：在 ONNX 导出时避免在 forward 函数内出现 if-else 分支，以确保静态计算图能正确构建并导出纯粹的特化模型。
    """
    original_init = cls.__init__

    def new_init(self, *args, **kwargs):
        # 执行原本的 __init__ 过程
        original_init(self, *args, **kwargs)

        if not hasattr(self, "task"):
            raise AttributeError("Instance must have 'task' attribute after __init__")

        chosen_method = None
        # 遍历类中所有方法，寻找那个携带有被 @register_forward 标记且等于期望 task 的方法
        for name in dir(cls):
            method = getattr(cls, name)
            if callable(method) and hasattr(method, "_forward_version"):
                if method._forward_version == self.task:
                    chosen_method = method
                    break

        if chosen_method is None:
            raise ValueError(f"No forward implementation for task: {self.task}")

        # 动态将挑选出的方法覆盖为当前对象的 forward() 方法
        self.forward = chosen_method.__get__(self, cls)

    cls.__init__ = new_init
    return cls


@select_forward
class _Onnx_GAEMimic_PolicyExporter(_OnnxPolicyExporter):
    """
    针对 SONIC（多模态/GAE拟合）架构专用的策略 ONNX 导出器。
    该类会根据传入任务类型（机器人硬件、人类动捕、三维关键点），剔除无用的输入配置，
    实现只导出“专门对口”那一模态的轻量化模型。
    """

    def __init__(
        self,
        actor_critic,
        env=None,
        normalizer=None,
        verbose=False,
        task: Literal["robot", "human", "keypoints"] = "robot",
    ):
        super().__init__(actor_critic, normalizer, verbose)

        self.task = task

        # 从策略网络中抽取三大不同模态输入特征的维度数
        assert (
            hasattr(actor_critic.actor, "actor_sk_dim")
            and hasattr(actor_critic.actor, "actor_sg_dim")
            and hasattr(actor_critic.actor, "actor_sh_dim")
        ), "Actor does not have keypoints, robot, and smplx dimensions."

        self.actor_sg_dim = actor_critic.actor.actor_sg_dim
        self.actor_sh_dim = actor_critic.actor.actor_sh_dim
        self.actor_sk_dim = actor_critic.actor.actor_sk_dim

        self.num_actions = actor_critic.actor.num_actions

        # 获取环境的所有观察向量名称与维度数
        all_obs_names = env.observation_manager.active_terms["policy"]
        group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]
        all_obs_dims = [dims[-1] for dims in group_obs_term_dim]

        # ---------------------------------------------------------
        # 根据不同模态，过滤掉不属于该模态的命令参数 (如：导出 robot 版就不该需要 human 的观测项)
        # ---------------------------------------------------------
        
        # 构建机器人端专属的观察维度
        self.robot_observation_names = []
        self.robot_observation_dims = []
        for i, name in enumerate(all_obs_names):
            if name not in ["human_command", "keypoints_command"]:
                self.robot_observation_names.append(name)
                self.robot_observation_dims.append(all_obs_dims[i])

        # 构建人体验证端专属的观察维度
        self.human_observation_names = []
        self.human_observation_dims = []
        for i, name in enumerate(all_obs_names):
            if name not in ["robot_command", "keypoints_command"]:
                self.human_observation_names.append(name)
                self.human_observation_dims.append(all_obs_dims[i])

        # 构建关键点追踪专属的观察维度
        self.keypoints_observation_names = []
        self.keypoints_observation_dims = []
        for i, name in enumerate(all_obs_names):
            if name not in ["robot_command", "human_command"]:
                self.keypoints_observation_names.append(name)
                self.keypoints_observation_dims.append(all_obs_dims[i])

    @register_forward("robot")
    def forward_robot(self, *args):
        """实机端使用的计算支路"""
        obs = torch.cat(args, dim=-1)
        robot_command = obs[:, : self.actor_sg_dim]          # 取出机器人指令段
        proprioceptive_state = obs[:, self.actor_sg_dim :]   # 取出余下的一般本体状态段
        return self.actor.forward_robot_exporter(robot_command, proprioceptive_state)

    @register_forward("human")
    def forward_human(self, *args):
        """人体重定向/拟合使用的计算支路"""
        obs = torch.cat(args, dim=-1)
        human_command = obs[:, : self.actor_sh_dim]
        proprioceptive_state = obs[:, self.actor_sh_dim :]
        return self.actor.forward_smplx_exporter(human_command, proprioceptive_state)

    @register_forward("keypoints")
    def forward_keypoints(self, *args):
        """关键点轨迹跟随使用的计算支路"""
        obs = torch.cat(args, dim=-1)
        keypoints_command = obs[:, : self.actor_sk_dim]
        proprioceptive_state = obs[:, self.actor_sk_dim :]
        return self.actor.forward_keypoints_exporter(keypoints_command, proprioceptive_state)

    def export(self, path, filename):
        """基于分离好的模态计算流和占位符(Dummy Inputs) 执行 ONNX 计算图编排与导出"""
        self.to("cpu")

        # 将当前所需要匹配的模态输入维度套现
        if self.task == "robot":
            self.observation_names = self.robot_observation_names
            self.observation_dims = self.robot_observation_dims
        elif self.task == "human":
            self.observation_names = self.human_observation_names
            self.observation_dims = self.human_observation_dims
        elif self.task == "keypoints":
            self.observation_names = self.keypoints_observation_names
            self.observation_dims = self.keypoints_observation_dims
        else:
            raise ValueError(f"Unknown task for GAEMimic exporter: {self.task}")

        # 使用上述分离出来的维度建立虚拟(Zeros)张量输入，ONNX 需要利用它们跑一次假推导来锁定模型拓扑结构
        dummy_inputs = []
        for dim in self.observation_dims:
            dummy_inputs.append(torch.zeros(1, dim))

        input_names = list(self.observation_names)

        # PyTorch内部提供的核心导出调用点
        torch.onnx.export(
            self,
            tuple(dummy_inputs),
            os.path.join(path, filename),
            export_params=True,             # 是否随同代码拓扑存入训练获取的权重
            opset_version=11,               # 推演用的ONNX算子协议版本
            verbose=self.verbose,
            input_names=input_names,        # 每项观测名称对应独立输口标记
            output_names=["actions"],       # 模型对外输口仅包含生成的控制参数
            dynamic_axes={},
        )


def list_to_csv_str(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
    """
    工具函数：将列表或数组转换为 CSV 格式的字符串，通常用于将结构化信息打包塞进 ONNX 属性中。
    如果是浮点数，默认保留 3 位小数。
    """
    fmt = f"{{:.{decimals}f}}"
    return delimiter.join(
        fmt.format(x) if isinstance(x, (int, float)) else str(x)
        for x in arr  # numbers → format, strings → as-is
    )


def attach_onnx_metadata(env: ManagerBasedRLEnv, run_path: str, path: str, filename="policy.onnx") -> None:
    """
    用于在已导出的 ONNX 模型中“注入”仿真环境的元数据（Metadata）。
    在 Sim-to-Real 的实机部署中，实机控制器需要知道对应训练环境的许多超参数设定，例如关节名称顺序、
    关节刚度 (stiffness)、阻尼 (damping)、动作缩放比例 (action_scale) 等。
    借由这个函数，部署端的 C++/Python 程序在加载 ONNX 时就能自动抽取恢复这些配置，免去了手动校对的麻烦与出错风险。
    """
    onnx_path = os.path.join(path, filename)

    # 获取策略网络所需的环境观察项名称
    observation_names = env.observation_manager.active_terms["policy"]
    observation_dims: list[int] = []  # Add observation dimensions

    # Get observation dimensions for each group
    # 提取每组观测项对应的特征维度尺寸
    group_obs_term_dim = env.observation_manager._group_obs_term_dim["policy"]  # list[list[int]]
    observation_dims = [dims[-1] for dims in group_obs_term_dim]

    # 整理要写入 ONNX 的元数据字典
    metadata = {
        "run_path": run_path,  # 模型对应的训练数据路径
        "joint_names": env.scene["robot"].data.joint_names,  # 机器人关节名称数组（用于在实机中对齐控制指令顺序）
        "body_names": env.scene["robot"].data.body_names,    # 机器人刚体部位名称数组
        "joint_stiffness": env.scene["robot"].data.joint_stiffness[0].cpu().tolist(),  # PD 伺服控制系统的 P 参数
        "joint_damping": env.scene["robot"].data.joint_damping[0].cpu().tolist(),      # PD 伺服控制系统的 D 参数
        "default_joint_pos": env.scene["robot"].data.default_joint_pos_nominal.cpu().tolist(), # 机器人的默认站立关节角度（零点/初始位姿）
        "command_names": env.command_manager.active_terms,   # 命令管理器当前激活的命令项
        "observation_names": observation_names,              # 各段状态观测空间的名称
        "observation_dims": observation_dims,  # Add to metadata # 各段状态空间的维度大小
        "action_scale": env.action_manager.get_term("joint_pos")._scale[0].cpu().tolist(), # 动作补偿在发往底层前被放大的缩放率 (Action Scale)
        "anchor_body_name": env.command_manager.get_term("motion").cfg.anchor_body_name,   # 运动追踪指定的基准躯体（通常是根节点如 pelvis/root）
        "tracking_body_names": env.command_manager.get_term("motion").cfg.body_names,      # 正在参与运动追踪匹配的其他身体部位名称
    }

    # 加载已存在的基础 ONNX 模型
    model = onnx.load(onnx_path)

    # 遍历元数据字典，将每个键值对转换为字符串原型后，插入到 ONNX 的 metadata props 列表中
    for k, v in metadata.items():
        entry = onnx.StringStringEntryProto()
        entry.key = k
        entry.value = list_to_csv_str(v) if isinstance(v, list) else str(v)
        model.metadata_props.append(entry)

    # 封存携带有完整附加物理与环境信息的全新 ONNX 模型文件
    onnx.save(model, onnx_path)
