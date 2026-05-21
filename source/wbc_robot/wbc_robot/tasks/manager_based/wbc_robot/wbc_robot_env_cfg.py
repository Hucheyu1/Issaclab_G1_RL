from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import wbc_robot.tasks.manager_based.wbc_robot.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # 地面：配置了平整地形 (plane)。给定了物理材质的摩擦系数 (static_friction=1.0, dynamic_friction=1.0) 以及贴图材质 (Shingles_01.mdl)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # 机器人 (robot): 被预留为 MISSING，意味着它需要由继承这个基类的特定机器人脚本（比如在 flat_env_cfg.py 中用 G1 实例去覆盖它）提供。
    robot: ArticulationCfg = MISSING
    # 光照 (light, sky_light): 添加了一个方向光和一个天光（穹顶光）提供仿真渲染的光源。
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    # 传感器 (contact_forces): 给机器人附加一个接触力传感器，设置记录的时间步 history_length=3，超过 10 力的接触会被认为是碰撞/接触
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##

# 指令（Commands）本质是强化学习目标在 Isaac Lab 中的表达。因为这是一个动作跟踪任务（Tracking/Mimic），所以所谓的“目标”主要是告诉机器人要在哪一拍摆什么姿势
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""
    # 定义允许机器人追踪动作命令及其初始位姿随机化的范围 pose_range 和 速度随机化范围 velocity_range。
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )

# 加入了数据集追踪能力。强制要求传入 dataset_dirs（数据集目录）和 splits。这就允许它从 100style 或者 lafan1 的 npz 数据集中提取连贯运动
@configclass
class MultiTracking_CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.MultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        # Dataset configuration
        dataset_dirs=MISSING,
        robot_name=MISSING,
        splits=MISSING,
    )


@configclass
class GAEMimic_CommandsCfg:
    """Command specifications for the MDP."""

    motion = mdp.GAEMimic_MultiMotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        # Dataset configuration
        dataset_dirs=MISSING,
        robot_name=MISSING,
        splits=MISSING,
    )

# 采用基于位置的 PD 控制。JointPositionActionCfg 表示网络直接输出目标关节角度（或偏移量），由底层的 PD 控制器将其映射为力矩。
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True, clip={".*": (-10.0, 10.0)}
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    # PolicyCfg (Actor输入): 网络可以“看到”给定的运动指令 (command)、根节点方向误差 (motion_anchor_ori_b)、基座角速度 (base_ang_vel)、
    # 各个关节的位置 (joint_pos) 和速度 (joint_vel)，以及上一步的动作 (actions)。并且为了 Domain Randomization（跨域泛化），对很多项加了均匀分布噪声 (Unoise)。
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(
            func=mdp.last_action, clip=(-10.0, 10.0)
        )  # NOTE bug actions should be clipped here as well to avoid large values

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
    # PrivilegedCfg (Critic输入): 这是在训练阶段（PPO 算法里）提供给 Value 网络的“上帝视角”。它不加噪声，并包含了所有的线速度、完全环境姿态、投影重力等。
    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(
            func=mdp.last_action, clip=(-10.0, 10.0)
        )  # NOTE bug actions should be clipped here as well to avoid large values

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

# 相比于前一种，这里的观测项加入了更丰富的引导信号，如过去几帧（以 interval=2 为步长的历史帧）的
# robot_command（机器人参考位姿）、human_command (SMPL位姿) 和 keypoints (环境关键点），用于加强强化学习动作生成。
@configclass
class GAEMimic_ObservationsCfg:
    """V3 robot_cmd, smplx_cmd, keypoints_cmd observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        robot_command = ObsTerm(
            func=mdp.motion_robot_joint_pos,
            params={
                "command_name": "motion",
                "interval": 2,
                "frames": 10,
            },
        )
        human_command = ObsTerm(
            func=mdp.motion_smplx_pose_body,
            params={
                "command_name": "motion",
                "interval": 2,
                "frames": 10,
            },
        )
        keypoints_command = ObsTerm(
            func=mdp.motion_keypoints_se3,
            params={
                "command_name": "motion",
                "interval": 2,
                "frames": 10,
            },
        )

        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(
            func=mdp.last_action, clip=(-10.0, 10.0)
        )  # NOTE bug actions should be clipped here as well to avoid large values

        def __post_init__(self):
            self.enable_corruption = True
            # 这行代码告诉物理引擎：“每一帧，请按从上到下的顺序收集这些数据，把它们像火腿肠一样首尾相连，拼接成一个长长的 1D 数组，然后丢给神经网络。”
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        robot_command = ObsTerm(
            func=mdp.motion_robot_joint_pos,
            params={
                "command_name": "motion",
                "interval": 2,
                "frames": 10,
            },
        )
        human_command = ObsTerm(
            func=mdp.motion_smplx_pose_body,
            params={
                "command_name": "motion",
                "interval": 2,
                "frames": 10,
            },
        )
        keypoints_command = ObsTerm(
            func=mdp.motion_keypoints_se3,
            params={
                "command_name": "motion",
                "interval": 2,
                "frames": 10,
            },
        )
        # 第一人称追踪误差 (Tracking Errors)
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(
            func=mdp.last_action, clip=(-10.0, 10.0)
        )  # NOTE bug actions should be clipped here as well to avoid large values

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()

# 为了实现 Zero-shot 部署到实体机器人上（Sim-to-Real），这里设置了强力的 Domain Randomization（随机化）：
@configclass
class EventCfg:
    """Configuration for events."""

    # mode="startup" (环境重置时): 随机化刚体摩擦系数、机器人关节初始位置偏移、躯干质心 (COM) 位置随机偏移（x/y/z ±10±10 cm），
    # 并将各环节的质量 (mass) 放缩到 0.8 ~ 1.2 倍。
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.6),
            "dynamic_friction_range": (0.1, 1.6),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )

    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # random_joint_friction = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "friction_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #     },
    # )

    # mode="interval" (仿真过程中): 定期（1到3秒之间）对机器人施加外力 (push_robot)，使机器人学会被推还能抗干扰
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # 鼓励机器人的整体位置、局部骨骼各环节部位以及动量/速度贴近参考运动（Command提供）；
    # 同时惩罚动作变化过快（action_rate_l2）、关节超出限制（joint_limit）以及不期望的接触（undesired_contacts）。
    motion_global_anchor_vel = RewTerm(
        func=mdp.motion_global_anchor_velocity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-5e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )

# 终止条件 (TerminationsCfg): 超时(time_out)、本体位置极度偏离参考轨迹导致跟不上 (anchor_pos, ee_body_pos) 
# 或者严重倾倒 (anchor_ori) 都会提前结束一个 episode (回合)
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )

# 课程学习 (CurriculumCfg): 根据特定的评价（在这里主要是 motion_global_anchor_ori 的表现表现好时），按步长提高环境的数据集抽样难度(adaptive_sampling_ratio)
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    adaptive_sampling_ratio = CurrTerm(
        func=mdp.adaptive_sampling_ratio,
        params={
            "reward_term_name": "motion_global_anchor_ori",
            "max_ratio": 0.8,
            "delta_ratio": 1e-1,
            "threshold": 0.9,
        },
    )


##
# Environment configuration
##


@configclass
class TrackingEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    
    # episode_length_s = 10.0 (一个Episode最多10秒)
    # dt = 0.005 (物理步长控制在 5ms，200Hz 控制频率)
    # decimation = 4 (网络策略层运行频率为 5毫秒 * 4 = 20毫秒，即 50Hz)
    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class MultiTracking_TrackingEnvCfg(TrackingEnvCfg):
    """Configuration for the locomotion multi-motion-tracking environment."""

    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands: MultiTracking_CommandsCfg = MultiTracking_CommandsCfg()


@configclass
class GAEMimic_TrackingEnvCfg(MultiTracking_TrackingEnvCfg):
    """Configuration for the locomotion multi-motion-tracking environment with GAE-Mimic observations."""

    def __post_init__(self):
        super().__post_init__()
        self.commands: GAEMimic_CommandsCfg = GAEMimic_CommandsCfg()
        self.observations: GAEMimic_ObservationsCfg = GAEMimic_ObservationsCfg()
