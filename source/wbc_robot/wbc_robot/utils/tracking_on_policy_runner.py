import os
from typing import Literal

import wandb
from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from wbc_robot.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class Tracking_OnPolicyRunner(OnPolicyRunner):
    """
    针对动作捕捉跟踪训练（Motion Tracking）量身定制的一个策略运行器（Runner）。
    它继承自 RSL-RL 库中标准的 `OnPolicyRunner` (通常用于执行 PPO)。
    最主要的改动在于，它重写了模型保存（save）的逻辑：在使用 wandb 监控训练的过程中，
    不仅会保存常见的 .pt (PyTorch模型权重) 检查点，还可以自动、实时地将策略转换为
    包含了物理环境元数据的专属 ONNX 模型，并自动上传给 WandB。
    这样一来，当实验还在跑的时候，研究人员就可以直接在控制台上下载当前最新版 ONNX 放到实机上无缝走测。
    """
    def __init__(
        self,
        env: VecEnv,
        train_cfg: dict,
        task_type: Literal["single_motion", "multi_motion", "gae_mimic"] = None,
        log_dir: str | None = None,
        device="cpu",
        registry_name: str = None,
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name
        self.task_type = task_type # 核心标志：记录当前训练的任务类型维度

    def save(self, path: str, infos=None):
        """
        重写 save 方法。
        不仅像原始方法那样保存 PyTorch 版本的 checkpoint，更额外地将模型编译为 ONNX，
        并且如果是 GAE Mimic (多模态) 任务，会智能分离出实机端 (robot)、数据处理端 (human)、追踪验证端 (keypoints) 三个纯净版的 ONNX 文件供不同链路独立测试提取。
        """
        # 1. 首先，正常的执行 RSL-RL 原生保存逻辑（保存 pth 权重、优化器状态等）
        super().save(path, infos)
        
        # 2. 如果开启了 Weights & Biases (wandb) 日志系统，则触发 ONNX 的动态导出并上传
        if self.logger_type in ["wandb"]:
            # 解析当前的保存路径格式以构造干净的导出文件名
            policy_path = path.split("model")[0]
            base_filename = policy_path.split("/")[-2]

            # 场景A：常规的单动作、多动作追踪
            if self.task_type in ["single_motion", "multi_motion"]:
                filename = base_filename + f"_{self.task_type}.onnx"
                # 利用我们在 exporter.py 中的核心导出函数输出 ONNX
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    self.alg.policy,
                    task_type=self.task_type,
                    normalizer=self.obs_normalizer,
                    path=policy_path,
                    filename=filename,
                )
                # 为该 ONNX 文件附着实机环境必备的元数据 (包含零点姿态、刚度阻尼、缩放率等)
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
                # 使用 wandb 的接口将生成好的带数据的 ONNX 手动推送到云端面板进行备份
                wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # 场景B：当前最前沿的多模态 SONIC/GAE 拟合算法
            elif self.task_type == "gae_mimic":
                # => 子任务 1/3: 导出专注于“实机端机器人关节信号驱动”的纯净计算版 (_robot.onnx)
                filename_robot = base_filename + "_robot.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    self.alg.policy,
                    task_type=self.task_type,
                    gaemimic_task="robot",
                    normalizer=self.obs_normalizer,
                    path=policy_path,
                    filename=filename_robot,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_robot)
                wandb.save(policy_path + filename_robot, base_path=os.path.dirname(policy_path))

                # => 子任务 2/3: 导出专注于“人体SMPL-X动捕驱动重定向”的计算版 (_human.onnx)
                filename_human = base_filename + "_human.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    self.alg.policy,
                    task_type=self.task_type,
                    gaemimic_task="human",
                    normalizer=self.obs_normalizer,
                    path=policy_path,
                    filename=filename_human,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_human)
                wandb.save(policy_path + filename_human, base_path=os.path.dirname(policy_path))

                # => 子任务 3/3: 导出专注验证“追踪3D零散关键点流”的纯净版 (_keypoints.onnx)
                filename_keypoints = base_filename + "_keypoints.onnx"
                export_motion_policy_as_onnx(
                    self.env.unwrapped,
                    self.alg.policy,
                    task_type=self.task_type,
                    gaemimic_task="keypoints",
                    normalizer=self.obs_normalizer,
                    path=policy_path,
                    filename=filename_keypoints,
                )
                attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename_keypoints)
                wandb.save(policy_path + filename_keypoints, base_path=os.path.dirname(policy_path))
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")
            
            # (可选) 将 Artifact (模型大注册表) 连接到本次 run，方便日后版本追踪溯源
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
