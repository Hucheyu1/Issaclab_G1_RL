"""用于从 NPZ 文件加载动作数据的动作数据集和数据加载器(Dataloader)。

该模块提供了基于 PyTorch 的 Dataset 和 DataLoader，用于从 NPZ 文件中
加载动作数据，并支持基于质量的分层采样以及训练/验证集切分。
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class Motion_Dataset(Dataset):
    """用于从 NPZ 文件加载动作数据的 PyTorch 数据集。

    该数据集采用延迟加载（在 __getitem__ 中按需加载），以避免
    一次性将所有动作数据加载到内存中可能导致的 OOM（内存溢出）问题。

    参数:
        dataset_dirs: 数据集目录路径列表。每个目录及其后续应遵循以下结构:
            ./datasets/npz_datasets/{dataset_name}/{robot_name}/
        robot_name: 机器人名称 (例如 "g1")。
        splits: 与每个 dataset_dir 对应的数据集切分列表。
            长度必须与 dataset_dirs 相同。每个元素可以是:
            - 字符串: 单个切分名称 (例如 "train", "val", "walk_subset")
            - 字符串列表: 需要合并的多个切分 (例如 ["train", "walk_subset"])

    示例:
        >>> # 每个数据集单个切分
        >>> dataset = Motion_Dataset(
        ...     dataset_dirs=["./datasets/npz_datasets/LAFAN1_Retargeting_Dataset"], robot_name="g1", splits=["train"]
        ... )
        >>> # 多个数据集和不同切分
        >>> dataset = Motion_Dataset(
        ...     dataset_dirs=[
        ...         "./datasets/npz_datasets/LAFAN1_Retargeting_Dataset",
        ...         "./datasets/npz_datasets/LAFAN1_Retargeting_Dataset",
        ...     ],
        ...     robot_name="g1",
        ...     splits=["train", ["train", "walk_subset"]],  # 第二个数据集合并了两个切分
        ... )
        >>> print(f"数据集大小: {len(dataset)}")
        >>> sample = dataset[0]
        >>> print(f"动作形状: {sample['joint_pos'].shape}")
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        robot_name: str,
        splits: list[str | list[str]],
        shuffle_seed: int = 42,
    ):
        """初始化 Motion_Dataset。

        参数:
            dataset_dirs: 数据集目录路径列表。
            robot_name: 机器人名称。
            splits: 数据集切分列表，长度必须与 dataset_dirs 相同。
                每个元素对应于 dataset_dirs 中同等索引的数据集。
                可以是字符串（单个切分）或字符串列表（要合并的多个切分）。

        引发:
            ValueError: 如果 splits 和 dataset_dirs 长度不同。
            FileNotFoundError: 如果数据集目录或信息文件（info.yaml/info.yml）不存在。
        """
        super().__init__()

        # 检查 splits 和 dataset_dirs 长度是否一致
        if len(splits) != len(dataset_dirs):
            raise ValueError(
                f"splits 的长度 ({len(splits)}) 必须匹配 dataset_dirs 的长度 ({len(dataset_dirs)})"
            )

        self.dataset_dirs = [Path(d).expanduser().resolve() for d in dataset_dirs]
        self.robot_name = robot_name
        self.splits = splits
        self.shuffle_seed = shuffle_seed

        # 用于存储 NPZ 文件路径和元数据
        self.npz_paths: list[Path] = []
        self.quantities: list[int] = []  # 每个动作片段的质量/难度评级
        self.motion_names: list[str] = []  # 每个动作的基础名称
        self.dataset_sources: list[str] = []  # 记录每个动作来自哪个数据集

        # 加载数据集信息并收集 NPZ 路径
        self._load_dataset_info()
        self._random_motions()

        print(f"[Motion_Dataset] 从 {len(self.dataset_dirs)} 个数据集中加载了 {len(self.npz_paths)} 个动作片段")
        print(f"[Motion_Dataset] 难度分布: {self._get_quantity_stats()}")

    def _load_dataset_info(self):
        """从 info.yaml 或 info.yml 文件加载数据集信息并收集 NPZ 路径。"""
        for dataset_idx, dataset_dir in enumerate(self.dataset_dirs):
            split_config = self.splits[dataset_idx]

            # 规范化 split_config 使其始终为列表
            if isinstance(split_config, str):
                split_names = [split_config]
            else:
                split_names = split_config

            # 仅尝试 YAML 文件（info.yaml 或 info.yml）
            info_path = None
            for ext in [".yaml", ".yml"]:
                candidate_path = dataset_dir / f"info{ext}"
                if candidate_path.exists():
                    info_path = candidate_path
                    break

            if info_path is None:
                raise FileNotFoundError(
                    f"在 {dataset_dir} 中未找到数据集信息文件。期望文件名: info.yaml 或 info.yml"
                )

            # 从 YAML 加载数据集信息
            with open(info_path) as f:
                info = yaml.safe_load(f)

            dataset_name = info["dataset"]

            # 处理配置中的每一个切分 (split)
            for split in split_names:
                split_info = info.get(split, {})

                if not split_info:
                    raise ValueError(f"[Motion_Dataset] {dataset_name} 中没有 '{split}' 数据")

                # 构建特定于机器人的 NPZ 文件夹路径
                robot_dir = dataset_dir / self.robot_name

                if not robot_dir.exists():
                    raise FileNotFoundError(f"未找到机器人目录: {robot_dir}")

                # 收集当前切分的 NPZ 路径
                for motion_name, quantity in split_info.items():
                    npz_path = robot_dir / f"{motion_name}.npz"

                    if npz_path.exists():
                        self.npz_paths.append(npz_path)
                        self.quantities.append(quantity)
                        self.motion_names.append(motion_name)
                        # 将来源记录为 dataset:split1+split2+... (针对合并切分的情况)
                        split_str = "+".join(split_names) if len(split_names) > 1 else split_names[0]
                        self.dataset_sources.append(f"{dataset_name}:{split_str}")
                    else:
                        print(f"[Motion_Dataset] 警告: 未找到 NPZ 文件: {npz_path}")

    def _random_motions(self):
        if self.shuffle_seed is not None:
            # 获取动作数量
            num_motions = len(self.npz_paths)
            # 使用固定随机数种子生成排列好的索引
            gen = torch.Generator().manual_seed(self.shuffle_seed)
            indices = torch.randperm(num_motions, generator=gen)
            self.shuffle_indices = indices.tolist()
        else:
            self.shuffle_indices = list(range(len(self.npz_paths)))

    def _get_quantity_stats(self) -> dict[int, int]:
        """获取难度分布统计信息。

        返回:
            将难度映射到数量的字典。
        """
        stats = {}
        for q in self.quantities:
            stats[q] = stats.get(q, 0) + 1
        return stats

    def __len__(self) -> int:
        """返回数据集中的动作片段数。

        返回:
            动作片段数。
        """
        return len(self.npz_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """加载并返回单个动作片段。

        此方法按需加载 NPZ 文件以避免内存问题。

        参数:
            idx: 要加载的动作片段的索引。

        返回:
            包含以下内容的字典:
                - motion: 包含动作数据的 numpy 数组字典:
                    - joint_pos: (num_frames, num_joints) 关节位置
                    - joint_vel: (num_frames, num_joints) 关节速度
                    - body_pos_w: (num_frames, num_bodies, 3) 刚体位置
                    - body_quat_w: (num_frames, num_bodies, 4) 刚体四元数
                    - body_lin_vel_w: (num_frames, num_bodies, 3) 刚体线速度
                    - body_ang_vel_w: (num_frames, num_bodies, 3) 刚体角速度
                - fps: 动作数据的每秒帧数
                - length: 动作的总帧数
                - duration: 动作持续时间 (秒)
                - npz_path: NPZ 文件路径
                - motion_name: 动作名称
                - quantity: 质量/难度评级 (1: 最好, 2: 中等, 3: 困难)
                - dataset_source: 来源数据集和切分 (格式: "dataset_name:split")
        """
        npz_path = self.npz_paths[self.shuffle_indices[idx]]

        # 加载动作数据
        data = np.load(npz_path)

        # 提取动作数据
        motion = {
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"],
            "body_quat_w": data["body_quat_w"],
            "body_lin_vel_w": data["body_lin_vel_w"],
            "body_ang_vel_w": data["body_ang_vel_w"],
        }

        # 如果可用，添加接触数据
        # if "contact" in data:
        # motion["contact"] = data["contact"]

        fps = int(data["fps"][0])
        length = motion["joint_pos"].shape[0]
        duration = length / fps

        return {
            "motion": motion,
            "fps": fps,
            "length": length,
            "duration": duration,
            "npz_path": str(npz_path),
            "motion_name": self.motion_names[idx],
            "quantity": self.quantities[idx],
            "dataset_source": self.dataset_sources[idx],
        }

    def get_motion_info(self) -> list[dict[str, Any]]:
        """获取所有动作的相关信息（不加载完整数据）。

        返回:
            包含动作元数据的字典列表。
        """
        info_list = []
        for i in range(len(self)):
            # 仅预加载以获取元数据（可优化为缓存）
            data = np.load(self.npz_paths[i])
            fps = int(data["fps"][0])
            length = data["joint_pos"].shape[0]

            info_list.append(
                {
                    "index": i,
                    "motion_name": self.motion_names[i],
                    "npz_path": str(self.npz_paths[i]),
                    "quantity": self.quantities[i],
                    "fps": fps,
                    "length": length,
                    "duration": length / fps,
                    "dataset_source": self.dataset_sources[i],
                }
            )

        return info_list

    def get_statistics(self) -> dict[str, Any]:
        """获取数据集统计信息。

        返回:
            包含数据集统计信息的字典。
        """
        total_frames = 0
        total_duration = 0.0
        lengths = []

        for i in range(len(self)):
            data = np.load(self.npz_paths[i])
            fps = int(data["fps"][0])
            length = data["joint_pos"].shape[0]
            duration = length / fps

            total_frames += length
            total_duration += duration
            lengths.append(length)

        return {
            "num_clips": len(self),
            "total_frames": total_frames,
            "total_duration": total_duration,
            "avg_frames_per_clip": total_frames / len(self) if len(self) > 0 else 0,
            "avg_duration_per_clip": total_duration / len(self) if len(self) > 0 else 0,
            "min_frames": min(lengths) if lengths else 0,
            "max_frames": max(lengths) if lengths else 0,
            "quantity_distribution": self._get_quantity_stats(),
        }


class Unify_Motion_Dataset(Motion_Dataset):
    """加载包含 SMPL-X 和关键点等扩展信息动作的数据集。

    继承自 Motion_Dataset 以访问经 extend_datasets.py 处理后的 NPZ
    文件中的拓展键值（如 SMPL-X 数据和机器人关键点 SE3 数据）。

    参数:
        dataset_dirs: 数据集目录路径列表
        robot_name: 机器人文件夹名称
        splits: 与每个 dataset_dir 对应的数据集切分列表
    """

    def __init__(
        self,
        dataset_dirs: list[str],
        robot_name: str,
        splits: list[str | list[str]],
    ) -> None:
        # 直接带着相同参数调用父类
        super().__init__(
            dataset_dirs=dataset_dirs,
            robot_name=robot_name,
            splits=splits,
        )
        print(f"[Unify_Motion_Dataset] 成功加载扩展动作数据集，共有 {len(self.npz_paths)} 个片段")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """从 NPZ 文件加载扩展的动作数据。

        重写了父类方法，当 NPZ 文件中存在额外的拓展键值
        (smplx_pose_body, robot_keypoints_trans 等) 时将其暴露。
        """
        npz_path = self.npz_paths[idx]

        # 加载动作数据
        data = np.load(npz_path)

        # 提取标准的动作数据 (与父类相同)
        motion = {
            "joint_pos": data["joint_pos"],
            "joint_vel": data["joint_vel"],
            "body_pos_w": data["body_pos_w"],
            "body_quat_w": data["body_quat_w"],
            "body_lin_vel_w": data["body_lin_vel_w"],
            "body_ang_vel_w": data["body_ang_vel_w"],
        }

        # 若可用则添加拓展键
        extended_keys = [
            "smplx_pose_body",
            "smplx_pose_body_global_rot",
            "robot_keypoints_trans",
            "robot_keypoints_rot",
        ]
        flatten_keys = set(extended_keys)

        for key in extended_keys:
            if key in data:
                arr = data[key]
                # 展平所有扩展键的最后两维
                if key in flatten_keys and arr.ndim >= 2:
                    arr = arr.reshape(arr.shape[0], -1)
                motion[key] = arr

        fps = int(data["fps"][0])
        length = motion["joint_pos"].shape[0]
        duration = length / fps

        return {
            "motion": motion,
            "fps": fps,
            "length": length,
            "duration": duration,
            "npz_path": str(npz_path),
            "motion_name": self.motion_names[idx],
            "quantity": self.quantities[idx],
            "dataset_source": self.dataset_sources[idx],
        }
