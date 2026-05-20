"""从 NPZ 文件加载和采样运动数据的运动数据加载器。

该模块提供了一个类似 PyTorch 的数据加载器，支持加权采样，
并在强化学习环境中为多运动跟踪提供高效的向量化批处理索引。
"""

import bisect
import math
from collections.abc import Sequence
from itertools import accumulate

import torch

from wbc_robot.utils.motion_dataset import Motion_Dataset, Unify_Motion_Dataset


class Motion_Dataloader:
    """用于采样运动片段的数据加载器，支持可选的加权采样。

    使用高效的拼接 + 偏移量索引以实现向量化的批处理访问。
    所有运动序列被拼接到单个张量中，并跟踪偏移量。

    参数:
        dataset: Motion_Dataset 实例
        device: 加载张量的设备

    示例:
        >>> dataset = Motion_Dataset(...)
        >>> dataloader = Motion_Dataloader(dataset)
        >>> # 通过全局索引直接访问缓冲区
        >>> motion_ids = torch.tensor([0, 1, 0, 2])
        >>> time_steps = torch.tensor([10, 20, 15, 5])
        >>> global_indices = dataloader.motion_offsets[motion_ids] + time_steps
        >>> joint_pos = dataloader.motion_buffer.joint_pos[global_indices]
        >>> # 均匀采样
        >>> indices = dataloader.sample(n=10)
        >>> # 加权采样
        >>> weights = compute_weights(...)
        >>> indices = dataloader.sample(n=10, weights=weights)
    """

    class MotionBuffer:
        """用于存储拼接后的运动数据张量的内部类。

        此类将所有运动数据封装在单个连续内存布局中，
        以实现高效的 GPU 加速批处理索引。

        属性:
            joint_pos: [总帧数, 关节数] - 关节位置
            joint_vel: [总帧数, 关节数] - 关节速度
            body_pos_w: [总帧数, 刚体数, 3] - 刚体位置 (世界坐标系)
            body_quat_w: [总帧数, 刚体数, 4] - 刚体四元数 (世界坐标系)
            body_lin_vel_w: [总帧数, 刚体数, 3] - 刚体线速度
            body_ang_vel_w: [总帧数, 刚体数, 3] - 刚体角速度
        """

        def __init__(self, body_indexes: Sequence[int]):
            """初始化空的运动缓冲区。"""
            self.joint_pos: torch.Tensor | None = None
            self.joint_vel: torch.Tensor | None = None
            self._body_pos_w: torch.Tensor | None = None
            self._body_quat_w: torch.Tensor | None = None
            self._body_lin_vel_w: torch.Tensor | None = None
            self._body_ang_vel_w: torch.Tensor | None = None
            self.body_indexes = body_indexes

        @property
        def body_pos_w(self) -> torch.Tensor:
            return self._body_pos_w[:, self.body_indexes]

        @property
        def body_quat_w(self) -> torch.Tensor:
            return self._body_quat_w[:, self.body_indexes]

        @property
        def body_lin_vel_w(self) -> torch.Tensor:
            return self._body_lin_vel_w[:, self.body_indexes]

        @property
        def body_ang_vel_w(self) -> torch.Tensor:
            return self._body_ang_vel_w[:, self.body_indexes]

    def __init__(
        self,
        dataset: Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
        enable_data_split: bool = False,
    ):
        """使用拼接的时间序列初始化数据加载器。

        参数:
            dataset: Motion_Dataset 实例
            body_indexes: 要提取的刚体索引
            device: 加载张量的设备
            world_size: 分布式进程的总数 (默认: 1 单进程)
            rank: 当前进程排名 (默认: 0)
            enable_data_split: 是否启用分布式数据分片 (默认: False)
        """
        self.dataset = dataset
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.enable_data_split = enable_data_split

        self._body_indexes = body_indexes

        # 初始化运动缓冲区
        self.motion_buffer = self.MotionBuffer(self._body_indexes)

        # 运动元数据 (将在 _preload_and_concatenate 中填充)
        self.motion_lengths: torch.Tensor  # [当前rank的运动数]，当前rank中每个运动的长度
        self.motion_offsets: torch.Tensor  # [当前rank的运动数]，每个运动的起始索引
        self.motion_fps: torch.Tensor  # [当前rank的运动数]，每个运动的 FPS
        self.time_step_total: int  # 当前rank缓冲区的总帧数
        self.num_motions: int  # 分配给当前rank的运动数量

        # 分布式数据跟踪
        self.start_motion_idx: int = 0  # 全局数据集中的起始运动索引
        self.end_motion_idx: int = 0  # 全局数据集中的结束运动索引
        self.global_num_motions: int = len(dataset)  # 数据集中的总运动数

        print(f"[Motion_Dataloader] Loading and concatenating motions for rank {self.rank}/{self.world_size}...")

        # 加载所有运动并将其拼接到单个张量中
        self._preload_and_concatenate()

        print(f"[Motion_Dataloader] Rank {self.rank} initialization complete. Total frames: {self.time_step_total}")

    def _preload_and_concatenate(self):
        """预加载所有运动并通过偏移量跟踪拼接到单个张量中。

        此方法预先加载所有运动数据，并沿时间维度拼接序列。
        每个运动的起始位置记录在 offsets 中。

        对于分布式训练：如果 enable_data_split=True，则会根据帧数划分各个 rank
        以平衡负载。

        内存高效：无需填充，仅存储原始数据。
        """
        # === 步骤 1: 加载所有运动的运动元数据 (暂不传输至 GPU) ===
        all_motion_lengths = []
        for i in range(self.global_num_motions):
            sample = self.dataset[i]
            all_motion_lengths.append(sample["length"])

        # === 步骤 2: 确定当前 rank 要加载的运动 ===
        if self.enable_data_split and self.world_size > 1:
            self._compute_rank_motion_indices(all_motion_lengths)
        else:
            # 单进程或禁用数据分片：加载所有运动
            self.start_motion_idx = 0
            self.end_motion_idx = self.global_num_motions

        self.num_motions = self.end_motion_idx - self.start_motion_idx

        # === 步骤 3: 仅加载并拼接当前 rank 的运动 ===
        data_lists = {
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }
        lengths = []
        fps_list = []

        # 加载分配给当前 rank 的运动
        for i in range(self.start_motion_idx, self.end_motion_idx):
            sample = self.dataset[i]
            motion_data = sample["motion"]

            # 添加到列表中
            for key in data_lists.keys():
                data_lists[key].append(torch.tensor(motion_data[key], dtype=torch.float32, device=self.device))

            lengths.append(sample["length"])
            fps_list.append(sample["fps"])

        # 连接当前 rank 缓冲区的所有序列
        self.motion_buffer.joint_pos = torch.cat(data_lists["joint_pos"], dim=0)
        self.motion_buffer.joint_vel = torch.cat(data_lists["joint_vel"], dim=0)
        self.motion_buffer._body_pos_w = torch.cat(data_lists["body_pos_w"], dim=0)
        self.motion_buffer._body_quat_w = torch.cat(data_lists["body_quat_w"], dim=0)
        self.motion_buffer._body_lin_vel_w = torch.cat(data_lists["body_lin_vel_w"], dim=0)
        self.motion_buffer._body_ang_vel_w = torch.cat(data_lists["body_ang_vel_w"], dim=0)

        # 计算当前 rank 的局部偏移量 (相对于当前 rank 的缓冲区)
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_offsets = torch.cat(
            [torch.tensor([0], device=self.device), torch.cumsum(self.motion_lengths, dim=0)[:-1]], dim=0
        )

        # 存储当前 rank 的 FPS
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)

        # 存储当前 rank 缓冲区的总长度
        self.time_step_total = self.motion_buffer.joint_pos.shape[0]

        print(f"[Motion_Dataloader] Rank {self.rank} concatenated tensors:")
        print(f"  - Motion indices: [{self.start_motion_idx}, {self.end_motion_idx})")
        print(f"  - Number of motions: {self.num_motions}")
        print(f"  - joint_pos: {self.motion_buffer.joint_pos.shape}")
        print(f"  - joint_vel: {self.motion_buffer.joint_vel.shape}")
        print(f"  - body_pos_w: {self.motion_buffer.body_pos_w.shape}")
        print(f"  - total_frames: {self.time_step_total}")
        print(f"  - motion_lengths range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]")

    def _compute_rank_motion_indices(self, all_motion_lengths: list[int]):
        """计算分配给当前 rank 的运动索引。

        策略：为 rank 分配完整的运动，使命每个 rank 获得大致相等的帧数。
        运动绝不会跨 rank 分割。保证跨所有 rank 是确定、非重叠且连续的范围。

        参数:
            all_motion_lengths: 数据集中所有运动的帧数列表
        """
        total_frames = sum(all_motion_lengths)
        target_frames_per_rank = math.ceil(total_frames / self.world_size) + 1  # 为了安全，在 world_size=1 时

        # cumsum
        cumulative_lengths = list(accumulate(all_motion_lengths))
        cur_rank_tg_start_frames = self.rank * target_frames_per_rank
        cur_rank_tg_end_frames = (self.rank + 1) * target_frames_per_rank

        # 查找运动索引
        start_motion_idx = bisect.bisect_left(cumulative_lengths.copy(), cur_rank_tg_start_frames)
        end_motion_idx = bisect.bisect_right(cumulative_lengths.copy(), cur_rank_tg_end_frames)
        self.start_motion_idx = start_motion_idx
        self.end_motion_idx = end_motion_idx

        rank_total_frames = sum(all_motion_lengths[i] for i in range(self.start_motion_idx, self.end_motion_idx))

        print(f"[Motion_Dataloader] Rank {self.rank}/{self.world_size} motion assignment:")
        print(f"  - Motion range: [{self.start_motion_idx}, {self.end_motion_idx})")
        print(f"  - Number of motions: {self.end_motion_idx - self.start_motion_idx}")
        print(f"  - Target frames per rank: {target_frames_per_rank}")
        print(f"  - Assigned frames: {rank_total_frames}")

    def get_motion_length(self, motion_id: int) -> int:
        """获取特定运动的长度。"""
        return self.motion_lengths[motion_id].item()

    def get_motion_fps(self, motion_id: int) -> float:
        """获取特定运动的 FPS。"""
        return self.motion_fps[motion_id].item()

    def sample(self, n: int, weights: torch.Tensor | list | None = None) -> torch.Tensor:
        """利用可选权重对 n 个运动索引进行采样。

        参数:
            n: 要采样的运动片段数量
            weights: 可选的 [num_motions] 权重张量或列表。
                    如果为 None，则使用均匀采样。
                    权重将在内部归一化。

        返回:
            motion_indices: 形状为 Tensor[n]，采样得到的数据集中的运动索引。

        示例:
            # 均匀采样
            indices = dataloader.sample(10)

            # 基于数量的加权采样
            weights = [0.85 if q==1 else 0.10 if q==2 else 0.05
                      for q in dataset.quantities]
            indices = dataloader.sample(10, weights=weights)

            # 自定义自适应采样
            weights = curriculum_weights * difficulty_scores * diversity_penalty
            indices = dataloader.sample(10, weights=weights)
        """
        if weights is None:
            # 均匀采样
            weights = torch.ones(self.num_motions, device=self.device)
        else:
            # 如果需要则转换为张量
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
            else:
                weights = weights.to(self.device)

            # 验证形状
            if weights.shape[0] != self.num_motions:
                raise ValueError(f"Weights shape mismatch: expected [{self.num_motions}], got {weights.shape}")

        # 确保权重为正数
        weights = torch.clamp(weights, min=1e-8)

        # 归一化
        weights = weights / weights.sum()

        # 采样
        motion_indices = torch.multinomial(weights, n, replacement=True)

        return motion_indices


class Unify_Motion_Dataloader(Motion_Dataloader):
    """配对的机器人和 SMPL-X 运动数据的扩展数据加载器。

    继承自 Motion_Dataloader 并添加对以下数据的支持:
    - smplx_pose_body: [总帧数, 126] - 扁平化的 SMPL-X 单个身体姿态
    - robot_keypoints_trans: [总帧数, 15] - 扁平化的机器人关键点平移
    - robot_keypoints_rot: [总帧数, 30] - 扁平化的 6D 机器人关键点旋转

    注意: 扩展键的数据在 Unify_Motion_Dataset.__getitem__() 中已经扁平化

    参数:
        dataset: Unify_Motion_Dataset 实例 (配对的机器人 + SMPL-X 数据)
        body_indexes: 用于过滤机器人刚体数据的索引
        device: 加载张量的设备

    示例:
        >>> dataset = Unify_Motion_Dataset(robot_map, smplx_map, robot_name="g1")
        >>> dataloader = Unify_Motion_Dataloader(dataset, body_indexes=[0, 1, 2, ...], device="cuda")
        >>> # 使用继承的采样方法
        >>> indices = dataloader.sample(n=32)
        >>> time_steps = torch.tensor([10, 20, 15, ...], device="cuda")
        >>> global_indices = dataloader.motion_offsets[indices] + time_steps
        >>> # 访问所有数据（包括扩展键的数据）
        >>> joint_pos = dataloader.motion_buffer.joint_pos[global_indices]
        >>> smplx_data = dataloader.motion_buffer.smplx_pose_body[global_indices]
        >>> keypoint_trans = dataloader.motion_buffer.robot_keypoints_trans[global_indices]
    """

    class UnifyMotionBuffer(Motion_Dataloader.MotionBuffer):
        """带有 SMPL-X 和扩展数据支持的扩展运动缓冲区。

        继承自父类的所有机器人运动缓冲区并新增：
        - smplx_pose_body: SMPL-X 身体姿态数据
        - robot_keypoints_trans: 机器人关键点平移
        - robot_keypoints_rot: 6D 的机器人关键点旋转
        """

        def __init__(self, body_indexes: Sequence[int]):
            super().__init__(body_indexes)
            self._smplx_pose_body = None
            self._robot_keypoints_trans = None
            self._robot_keypoints_rot = None

        @property
        def smplx_pose_body(self) -> torch.Tensor | None:
            """SMPL-X 6D 形式的身体姿态数据。"""
            return self._smplx_pose_body

        @property
        def robot_keypoints_trans(self) -> torch.Tensor | None:
            """机器人关键点 SE3 平移数据。"""
            return self._robot_keypoints_trans

        @property
        def robot_keypoints_rot(self) -> torch.Tensor | None:
            """机器人关键点 6D 形式中的 SE3 旋转数据。"""
            return self._robot_keypoints_rot

    def __init__(
        self,
        dataset: Unify_Motion_Dataset,
        body_indexes: Sequence[int],
        device: str = "cuda",
        world_size: int = 1,
        rank: int = 0,
        enable_data_split: bool = False,
    ):
        """初始化 Unify_Motion_Dataloader。

        参数:
            dataset: 含有配对数据的 Unify_Motion_Dataset 实例
            body_indexes: 刚体过滤索引序列
            device: 加载张量的设备
            world_size: 分布式进程的总数 (默认: 1 单进程)
            rank: 当前进程排名 (默认: 0)
            enable_data_split: 是否启用分布式数据分片 (默认: False)
        """
        self.dataset = dataset
        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.enable_data_split = enable_data_split

        self._body_indexes = body_indexes

        # 使用扩展的缓冲区类
        self.motion_buffer = self.UnifyMotionBuffer(self._body_indexes)

        # 运动元数据 (将在 _preload_and_concatenate 中填充)
        self.motion_lengths: torch.Tensor
        self.motion_offsets: torch.Tensor
        self.motion_fps: torch.Tensor
        self.time_step_total: int
        self.num_motions: int

        # 分布式数据跟踪
        self.start_motion_idx: int = 0  # 全局数据集中的起始运动索引
        self.end_motion_idx: int = 0  # 全局数据集中的结束运动索引
        self.global_num_motions: int = len(dataset)  # 数据集中的总运动数

        print(f"[Unify_Motion_Dataloader] Loading and concatenating motions for rank {self.rank}/{self.world_size}...")

        # 加载所有运动并拼接
        self._preload_and_concatenate()

        print(
            f"[Unify_Motion_Dataloader] Rank {self.rank} initialization complete. Total frames: {self.time_step_total}"
        )

    def _preload_and_concatenate(self) -> None:
        """预加载所有配对的运动并进行含有扩展数据处理的拼接。

        扩展了父类方法以处理 SMPL-X 以及机器人扩展数据。
        对于分布式训练：如果 enable_data_split=True，则会基于帧数
        跨多个 rank 分配运动以平衡负载。
        """
        # === 步骤 1: 加载所有运动的运动元数据 (暂不传输至 GPU) ===
        all_motion_lengths = []
        for i in range(self.global_num_motions):
            sample = self.dataset[i]
            all_motion_lengths.append(sample["length"])

        # === 步骤 2: 确定当前 rank 要加载的运动 ===
        if self.enable_data_split and self.world_size > 1:
            self._compute_rank_motion_indices(all_motion_lengths)
        else:
            # 单进程或禁用数据分片：加载所有运动
            self.start_motion_idx = 0
            self.end_motion_idx = self.global_num_motions

        self.num_motions = self.end_motion_idx - self.start_motion_idx

        # === 步骤 3: 仅加载并拼接当前 rank 的运动 ===
        # 基础数据列表 (继承自父类)
        data_lists = {
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }

        # 用于 SMPL-X 和机器人关键点的扩展数据列表
        # 注意: 数据由 Unify_Motion_Dataset 经过了扁平化处理
        extended_lists = {
            "smplx_pose_body": [],
            "robot_keypoints_trans": [],
            "robot_keypoints_rot": [],
        }

        lengths = []
        fps_list = []

        # 加载分配给当前 rank 的运动
        for i in range(self.start_motion_idx, self.end_motion_idx):
            item = self.dataset[i]
            robot_motion = item["motion"]
            robot_len = item["length"]
            robot_item = item

            # 加载基础数据 (复用 Motion_Dataset 接口)
            for key in data_lists.keys():
                data_lists[key].append(torch.tensor(robot_motion[key], dtype=torch.float32, device=self.device))

            # 加载扩展数据 (必须经过 extend_datasets.py 预处理生成)
            # 注意: 数据由 Unify_Motion_Dataset 经过了扁平化处理
            for ext_key in extended_lists.keys():
                extended_lists[ext_key].append(
                    torch.tensor(robot_motion[ext_key], dtype=torch.float32, device=self.device)
                )

            lengths.append(robot_len)
            fps_list.append(robot_item["fps"])

        # 拼接基础数据 (机器人运动)
        self.motion_buffer.joint_pos = torch.cat(data_lists["joint_pos"], dim=0)
        self.motion_buffer.joint_vel = torch.cat(data_lists["joint_vel"], dim=0)
        self.motion_buffer._body_pos_w = torch.cat(data_lists["body_pos_w"], dim=0)
        self.motion_buffer._body_quat_w = torch.cat(data_lists["body_quat_w"], dim=0)
        self.motion_buffer._body_lin_vel_w = torch.cat(data_lists["body_lin_vel_w"], dim=0)
        self.motion_buffer._body_ang_vel_w = torch.cat(data_lists["body_ang_vel_w"], dim=0)

        # 拼接扩展数据 (必须经过 extend_datasets.py 预处理生成)
        self.motion_buffer._smplx_pose_body = torch.cat(extended_lists["smplx_pose_body"], dim=0)
        self.motion_buffer._robot_keypoints_trans = torch.cat(extended_lists["robot_keypoints_trans"], dim=0)
        self.motion_buffer._robot_keypoints_rot = torch.cat(extended_lists["robot_keypoints_rot"], dim=0)

        # 计算运动元数据
        self.motion_lengths = torch.tensor(lengths, dtype=torch.long, device=self.device)
        self.motion_offsets = torch.cat(
            [torch.tensor([0], dtype=torch.long, device=self.device), torch.cumsum(self.motion_lengths, dim=0)[:-1]],
            dim=0,
        )
        self.motion_fps = torch.tensor(fps_list, dtype=torch.float32, device=self.device)
        self.time_step_total = self.motion_buffer.joint_pos.shape[0]

        # 打印缓冲区信息
        print(f"[Unify_Motion_Dataloader] Rank {self.rank} concatenated tensors:")
        print(f"  - Motion indices: [{self.start_motion_idx}, {self.end_motion_idx})")
        print(f"  - Number of motions: {self.num_motions}")
        print(f"  joint_pos: {self.motion_buffer.joint_pos.shape}")
        print(f"  joint_vel: {self.motion_buffer.joint_vel.shape}")
        print(f"  body_pos_w: {self.motion_buffer.body_pos_w.shape}")
        print(f"  smplx_pose_body: {self.motion_buffer._smplx_pose_body.shape}")
        print(f"  robot_keypoints_trans: {self.motion_buffer._robot_keypoints_trans.shape}")
        print(f"  robot_keypoints_rot: {self.motion_buffer._robot_keypoints_rot.shape}")
        print(f"  total_frames: {self.time_step_total}")
        print(
            f"  motion_lengths: {self.motion_lengths.shape}, range: [{self.motion_lengths.min()}, {self.motion_lengths.max()}]"
        )
        print(f"  motion_offsets: {self.motion_offsets.shape}")
