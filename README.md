# Overview

## 📋 可用环境总览 (Available Environments)
```bash
python scripts/tools/list_envs.py
```
| 序号 | 任务名称 (Task Name) | 入口点 (Entry Point) | 配置文件 (Config) |
| :--- | :--- | :--- | :--- |
| 1 | `Template-Tracking-Flat-G1-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:G1FlatEnvCfg` |
| 2 | `Template-MultiTracking-Flat-G1-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:MultiTracking_G1FlatEnvCfg` |
| 3 | `Template-GAEMimic-Flat-G1-v0` | `isaaclab.envs:ManagerBasedRLEnv` | `wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:GAEMimic_G1FlatEnvCfg` |
| 4 | `Template-GAEMimic-Flat-G1-v1` | `isaaclab.envs:ManagerBasedRLEnv` | `wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:GAEMimic_G1FlatEnvCfg` |

## 🚀 快速开始 (Quick Start)

### 1. 数据集准备与安装
首先需要克隆所需的机器人模型和动作数据集和修改后的强化学习库：

```bash
# 下载 Unitree G1 机器人模型
git clone [https://huggingface.co/datasets/unitreerobotics/unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model)

# 安装 Git LFS 并下载 GAE Mimic 动作数据集
git lfs install
git clone [https://www.modelscope.cn/datasets/seulzx/gae_mimic_dataset.git](https://www.modelscope.cn/datasets/seulzx/gae_mimic_dataset.git)

# 安装修改后的rsl_rl（使用actor_critic_triple_ae与triple_ae_ppo）
cd rsl_rl
python -m pip install -e .
```
### 2. 数据集脚本工具说明

**数据集目录结构：**
*   `original_datasets`：原始人类数据集
*   `extend_datasets`：扩展/重定向后的机器人数据集
    *   `lafan1_dataset`：基础动作与平滑过渡
    *   `100style_dataset`：性格与风格化运动
    *   `omomo_dataset`：复杂的全身物理交互

**数据处理脚本集（位于 `scripts/data/`）：**

*   **`csv_to_npz.py` / `offline_csv_to_npz.py`**
    *   **作用：** 将单个记录了关节角度的 `.csv` 文件，打包压缩成体积更小、读取速度快百倍的 `.npz` 二进制文件。
*   **`gmr_pkl_to_csv.py`**
    *   **作用：** GMR（动作重定向）算法跑完后，通常会输出 Python 的 `.pkl`（Pickle）字典文件。这个脚本负责把它解包，并展平保存为人类可读的 `.csv` 文件，作为后续处理的第一步。
*   **`smplx_interpolate.py`**
    *   **作用：** 帧率插值（极其重要！）。网上下载的人类视频可能是 30帧/秒 或 60帧/秒，但 RL 物理引擎（Isaac Lab）控制频率可能固定在 50Hz（每 0.02 秒一步）。这个脚本利用数学插值算法，把人类动作的时间轴强行对齐到机器人的控制时间轴上，防止动作变快或变慢。当你有成百上千个动作片段时，这实现了自动化转换。
*   **`offline_csv_to_npz_datasets.py`**
    *   **作用：** 一键遍历整个数据集文件夹，把里面所有的 CSV 批量转换为 NPZ。
*   **`parallel_offline_csv_to_npz_datasets.py`**
    *   **作用：** `offline_csv_to_npz_datasets.py` 的多进程加速版。利用电脑的多个 CPU 核心同时干活，把处理数据集的时间从几个小时缩短到几分钟。
*   **`auto_datasets_yaml.py`**
    *   **作用：** 生成数据集所需的 `info.yaml` 文件。它会自动扫描生成的所有 `.npz` 动作片段，统计它们的长度、数量，并写成配置文件，供 `train.py` 里的 `MultiMotionCommand` 读取使用。
*   **`extend_datasets.py`**
    *   **作用：** 负责整理目录结构，把处理好的数据挪到最终的 `extend_datasets` 文件夹中。
*   **`csv_to_video.py`**
    *   **作用：** 读取 `.csv` 轨迹，并利用简单的渲染器（比如 matplotlib 3D 或者 pybullet）快速画出一个火柴人/机器人模型，导出为 `.mp4` 视频供预览。
*   **`replay_npz.py`**
    *   **作用：** 读取已经打包好的 `.npz` 最终文件，弹出一个 3D 窗口实时回放机器人的动作序列。这是训练前检查数据正确性的最后一道防线。
*   **`upload_npz.py`**
    *   **作用：** 辅助脚本，如果你在本地处理好了数据集，但要在云端 GPU 服务器（如 gpufree 容器）上训练，这个脚本可以一键把几 GB 的 `.npz` 数据集传到云服务器或对象存储（如 AWS S3、阿里云 OSS）里。
```bash
# 回放动作
python scripts/replay_npz.py --motion_file datasets/extend_datasets/lafan1_dataset/g1/train/dance1_subject1.npz
```
### 3. 训练与测试命令
```bash
# 全新训练 (无头模式)
python scripts/rsl_rl/train.py --task Template-MultiTracking-Flat-G1-v0 --headless
python scripts/rsl_rl/train.py --task Template-GAEMimic-Flat-G1-v0 --headless
# 恢复训练 (加载指定的历史 Checkpoint)
python scripts/rsl_rl/train.py --task Template-MultiTracking-Flat-G1-v0 --headless --resume --load_run 2026-05-16_20-58-44
# 播放与测试 (可视化渲染，设置录制 1000 帧)
python scripts/rsl_rl/play.py --task Template-MultiTracking-Flat-G1-v0 --headless --video --video_length 1000 --num_envs 1
python scripts/rsl_rl/play.py --task Template-MultiTracking-Flat-G1-v0 --num_envs 1 --motion_file datasets/extend_datasets/lafan1_dataset/g1/train/dance1_subject1.npz
```

### 4. 查看训练日志
```bash
tensorboard --logdir /root/gpufree-data/lab_lecture/wbc_robot/logs/rsl_rl/multi_g1_flat/
tensorboard --logdir /root/gpufree-data/lab_lecture/wbc_robot/logs/rsl_rl/gaemimic_g1_flat/
```

## ⚙️ MultiTracking 环境配置 (MDP Environment Setup)

### 5.1 动作、命令与事件 (Actions, Commands & Events)
| 类别 | 模块项 | 详情描述 |
| :--- | :--- | :--- |
| **动作 (Action)** | `joint_pos` | 29 维，输出给 G1 机器人的各个关节目标位置。 |
| **任务目标 (Command)** | `motion` | `MultiMotionCommand`（多动作模仿模式的控制指令）。 |
| **初始化事件 (Event)** | `physics_material`, `add_joint_default_pos`, `base_com`, `body_mass` | 在环境刚启动 (startup) 时随机设置物理属性，用于 Domain Randomization。 |
| **周期扰动 (Event)** | `push_robot` | 每隔 1.0~3.0 秒给机器人一个外力推力，增加其行走的鲁棒性。 |
| **课程学习 (Curriculum)** | `adaptive_sampling_ratio` | 动态调整采样难度，遇到不擅长的动作会增加被采样的概率。 |

### 5.2 观测空间 (Observations)

| 类别 | 维度 | 包含内容与解释 |
| :--- | :--- | :--- |
| **Actor (Policy)** <br>*(机器人实际能看见的)* | **154 维** | **`command`** (58维): 核心追踪指令，通常包含了未来参考帧的期望状态（目标关节角度、速度等）。<br>**`base_ang_vel`** (3维): 躯干的 Roll/Pitch/Yaw 旋转速度（来自IMU）。<br>**`motion_anchor_ori_b`** (6维): 目标相对姿态。影子参考对象（Anchor）相对于 G1 躯干的朝向偏差。<br>**`joint_pos`** (29维), **`joint_vel`** (29维): 各个关节的当前位置和速度。<br>**`actions`** (29维): 机器人上一帧采取的动作。 |
| **Critic (Value)** <br>*(评估用的“上帝视角”)* | **289 维** | *包含 Actor 的所有信息，并额外增加了：*<br>**`projected_gravity`** (3维): 投影重力，将世界坐标系下向下的重力向量通过矩阵投影到机器人倾斜的躯干坐标系中，让机器人立刻感知前倾后仰状态。<br>**`base_lin_vel`** (3维): 真实绝对移动速度（通常在现实中很难准确获得，因此仅供 Critic）。<br>**`motion_anchor_pos_b`** (3维): 数据集里影子相对机器人中心的三维距离偏差 (x, y, z)。<br>**`body_pos`** (42维): 14个刚体连杆的三维位置 (14×3)，极度精准掌握四肢绝对空间。<br>**`body_ori`** (84维): 14个刚体连杆，每个连杆用 6D 连续旋转向量表示姿态 (14×6)。 |

### 5.3 奖励函数设定 (Rewards)

| 奖励项 | 权重 (Weight) | 描述 (Description) |
| :--- | :--- | :--- |
| `motion_body_pos` / `ori` | **+ 1.0** | *(主要奖励)* 鼓励躯干和身体各部位精准追踪目标位置和姿态。 |
| `motion_body_lin_vel` / `ang_vel` | **+ 1.0** | *(主要奖励)* 鼓励匹配目标动作的全局线速度和角速度。 |
| `motion_global_anchor_vel` / `ori` | **+ 0.5** | *(次要奖励)* 鼓励全局锚点(Anchor)追踪。 |
| `undesired_contacts` | **- 0.1** | *(轻微惩罚)* 防止发生不该有的物理碰撞（如手腕碰地、躯干摔倒）。 |
| `action_rate_l2` | **- 0.5** | *(中度惩罚)* 防止动作输出突变，保证电机平滑发力。 |
| `joint_limit` | **- 10.0** | *(极其严厉)* 严禁关节突破物理限位，保护真机安全。 |

## ⚙️ Deploy
### Setup
```bash
# 安装 mujoco https://github.com/google-deepmind/mujoco/releases
mkdir -p ~/.mujoco && tar -zxvf mujoco-3.8.0-linux-x86_64.tar.gz -C ~/.mujoco

# C++ Simulator (simulate)
sudo apt install libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2/
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF # Install on the /usr/local directory
sudo make install

# Python Simulator (simulate_python) (可选) https://github.com/unitreerobotics/unitree_mujoco?tab=readme-ov-file#installation
cd ~
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
pip install mujoco
pip install pygame

# Compile unitree_mujoco
git clone https://github.com/unitreerobotics/unitree_mujoco.git
cd unitree_mujoco/simulate/
ln -s ~/.mujoco/mujoco-3.8.0 mujoco
cd unitree_mujoco/simulate
mkdir build && cd build
cmake ..
make -j4

# Compile the robot_controller
cd deploy/robots/g1_29dof
mkdir build && cd build
cmake .. && make
```

### Sim2Sim
```bash
# Set the robot at /simulate/config.yaml to g1
# Set domain_id to 0
# Set enable_elastic_hand to 1
# Set use_joystck to 1.
# start simulation
cd unitree_mujoco/simulate/build
./unitree_mujoco
# ./unitree_mujoco -i 0 -n eth0 -r g1 -s scene_29dof.xml # alternative
cd unitree_rl_lab/deploy/robots/g1_29dof/build
./g1_ctrl
# 1. press [L2 + Up] to set the robot to stand up
# 2. Click the mujoco window, and then press 8 to make the robot feet touch the ground.
# 3. Press [R1 + X] to run the policy.
# 4. Click the mujoco window, and then press 9 to disable the elastic band.
```

