# Copyright (c) 2024 KScale Labs.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Velocity-Rough-StompyMini-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.StompyMiniRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyMiniRoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Rough-StompyMini-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.StompyMiniRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyMiniRoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-StompyMini-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.StompyMiniFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyMiniFlatPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-StompyMini-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.StompyMiniFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyMiniFlatPPORunnerCfg,
    },
)
