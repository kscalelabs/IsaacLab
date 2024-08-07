# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Velocity-Rough-Stompy-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.StompyRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyRoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Stompy-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.StompyRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyRoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Stompy-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.StompyFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyFlatPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Stompy-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.StompyFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.StompyFlatPPORunnerCfg,
    },
)
