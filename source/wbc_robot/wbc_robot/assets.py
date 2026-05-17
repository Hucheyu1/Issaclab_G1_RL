import os

# 1. 获取当前文件 (assets.py) 所在的绝对路径
# 此时路径为： /.../WBC_ROBOT/source/wbc_robot/wbc_robot
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 向上一层一层退，退回到 WBC_ROBOT 的项目根目录
# 返回 3 层：wbc_robot -> wbc_robot (包名) -> source -> WBC_ROBOT
PROJECT_ROOT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, "..", "..", ".."))

# 3. 将 ASSET_DIR 指向你放在根目录的 unitree_model 文件夹
# ASSET_DIR = os.path.join(PROJECT_ROOT_DIR, "unitree_model")
ASSET_DIR = PROJECT_ROOT_DIR
# 顺便打印一下，方便你在终端里排查路径对不对（可选）
# print(f"[INFO] 机器人资产目录已指向: {ASSET_DIR}")
