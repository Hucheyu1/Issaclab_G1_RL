# Template for Isaac Lab Projects

+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
|                                                              Available Environments in Isaac Lab                                                               |
+--------+-----------------------------------+---------------------------------+---------------------------------------------------------------------------------+
| S. No. | Task Name                         | Entry Point                     | Config                                                                          |
+--------+-----------------------------------+---------------------------------+---------------------------------------------------------------------------------+
|   1    | Template-Tracking-Flat-G1-v0      | isaaclab.envs:ManagerBasedRLEnv | wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:G1FlatEnvCfg               |
|   2    | Template-MultiTracking-Flat-G1-v0 | isaaclab.envs:ManagerBasedRLEnv | wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:MultiTracking_G1FlatEnvCfg |
|   3    | Template-GAEMimic-Flat-G1-v0      | isaaclab.envs:ManagerBasedRLEnv | wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:GAEMimic_G1FlatEnvCfg      |
|   4    | Template-GAEMimic-Flat-G1-v1      | isaaclab.envs:ManagerBasedRLEnv | wbc_robot.tasks.manager_based.wbc_robot.flat_env_cfg:GAEMimic_G1FlatEnvCfg      |
+--------+-----------------------------------+---------------------------------+---------------------------------------------------------------------------------+

git clone https://huggingface.co/datasets/unitreerobotics/unitree_model

git lfs install

git clone https://www.modelscope.cn/datasets/seulzx/gae_mimic_dataset.git

python scripts/rsl_rl/train.py --task Template-MultiTracking-Flat-G1-v0 --headless
python scripts/rsl_rl/train.py --task Template-MultiTracking-Flat-G1-v0 --headless --resume --load_run 2026-05-16_20-58-44

python scripts/rsl_rl/play.py --task Template-MultiTracking-Flat-G1-v0 --headless --video --video_length 1000 --num_envs 1

tensorboard --logdir /root/gpufree-data/lab_lecture/wbc_robot/logs/rsl_rl/multi_g1_flat/


Action (动作指令),joint_pos,29 维（输出给 G1 关节的目标位置）
Command (任务目标),motion,MultiMotionCommand（多动作模仿模式）
Event (初始化事件),"physics_material, add_joint_default_pos, base_com, body_mass",在环境刚启动 (startup) 时设置物理属性
Event (周期扰动),push_robot,每隔 1.0~3.0 秒给机器人一个推力，增加鲁棒性
Curriculum (课程学习),adaptive_sampling_ratio,动态调整数据采样难度


Actor (Policy)机器人实际能看到的,154 维,"command (58), motion_anchor_ori_b (6) base_ang_vel (3), joint_pos (29)joint_vel (29), actions (29)"
Critic (Value)训练时评估用的“上帝视角”,289 维,"包含 Actor 的所有信息，额外增加了：motion_anchor_pos_b (3), body_pos (42)body_ori (84), base_lin_vel (3)projected_gravity (3)"

command (58)：核心追踪指令。在你的 MultiTracking 环境中，这通常包含了从 lafan1 数据集中切出来的未来参考帧的期望状态（比如目标关节角度、目标根节点速度等加起来刚好 58 维）。
base_ang_vel: 躯干角速度,机器人骨盆/躯干在 Roll（翻滚）、Pitch（俯仰）、Yaw（偏航）三个方向的旋转速度，数据直接来自 IMU 陀螺仪
motion_anchor_ori_b (6)：目标相对姿态。代表数据集里那个“影子机器人”（Anchor）的朝向，相对于 G1 当前躯干（Base）的偏差

base_lin_vel (3)：躯干线速度。躯干在 x, y, z 三个方向的真实绝对移动速度。（现实中很难通过机载传感器算准）。
projected_gravity (3)：投影重力。这是足式机器人 RL 中最经典、最伟大的技巧之一！它将世界绝对坐标系下向下的重力向量 [0, 0, -9.81]，通过数学矩阵投影到机器人倾斜的躯干坐标系中。
意义：机器人不需要知道绝对的四元数姿态，它只要感受一下这 3 个重力投影分量，就能立刻知道自己目前是前倾、后仰还是平躺，从而快速做出平衡反应。
motion_anchor_pos_b (3)：目标相对位置。数据集里那个“影子”相对机器人中心的三维距离偏差 (x, y, z)
body_pos (42)：全身各连杆的三维位置。G1 机器人被切分成了 14 个刚体连杆（比如大腿、小腿、脚掌、上臂等，14 × 3 = 42）。这能让 Critic 极其精准地掌握四肢在空间中的位置。
body_ori (84)：全身各连杆的姿态。同理，14 个刚体连杆，每个连杆用 6D 连续旋转向量表示姿态（14 × 6 = 84）。


motion_body_pos / ori,+ 1.0,(主要奖励) 鼓励躯干和身体各部位精准追踪目标姿态
motion_body_lin_vel / ang_vel,+ 1.0,(主要奖励) 鼓励匹配目标动作的线速度和角速度
motion_global_anchor_vel / ori,+ 0.5,(次要奖励) 鼓励全局锚点追踪
undesired_contacts,- 0.1,(轻微惩罚) 防止发生不该有的物理碰撞
action_rate_l2,- 0.5,(中度惩罚) 防止动作输出突变，保证电机平滑发力
joint_limit,- 10.0,(极其严厉) 严禁关节突破物理限位，保护真机安全

## Overview

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/wbc_robot

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/wbc_robot/wbc_robot/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/wbc_robot"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```