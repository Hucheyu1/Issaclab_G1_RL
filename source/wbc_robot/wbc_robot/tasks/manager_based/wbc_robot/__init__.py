import gymnasium as gym

from . import agents

# 不需要在这里显式导入具体的类了，只需要让 Gym 知道路径即可
# 也就是说，下面这几行可以删掉或者注释掉：
# from . import agents, flat_env_cfg
# from .agents import rsl_rl_ppo_cfg

##
# Register Gym environments.
##

gym.register(
    id="Template-Tracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-MultiTracking-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:MultiTracking_G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MultiTracking_G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-GAEMimic-Flat-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:GAEMimic_G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GAEMimic_G1FlatPPORunnerCfg",
    },
)

gym.register(
    id="Template-GAEMimic-Flat-G1-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:GAEMimic_G1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:GAEMimic_Large_G1FlatPPORunnerCfg",
    },
)

# import gymnasium as gym

# from . import agents, flat_env_cfg
# from .agents import rsl_rl_ppo_cfg

# ##
# # Register Gym environments.
# ##

# gym.register(
#     id="Template-Tracking-Flat-G1-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.G1FlatEnvCfg,
#         "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.G1FlatPPORunnerCfg,
#     },
# )


# gym.register(
#     id="Template-MultiTracking-Flat-G1-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.MultiTracking_G1FlatEnvCfg,
#         "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.MultiTracking_G1FlatPPORunnerCfg,
#     },
# )

# gym.register(
#     id="Template-GAEMimic-Flat-G1-v0",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.GAEMimic_G1FlatEnvCfg,
#         "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.GAEMimic_G1FlatPPORunnerCfg,
#     },
# )

# gym.register(
#     id="Template-GAEMimic-Flat-G1-v1",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": flat_env_cfg.GAEMimic_G1FlatEnvCfg,
#         "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.GAEMimic_Large_G1FlatPPORunnerCfg,
#     },
# )
