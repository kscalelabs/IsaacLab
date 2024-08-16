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
    id="Isaac-Velocity-Rough-Dora-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.DoraRoughEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DoraRoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Rough-Dora-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.DoraRoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DoraRoughPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Dora-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DoraFlatEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DoraFlatPPORunnerCfg,
    },
)


gym.register(
    id="Isaac-Velocity-Flat-Dora-Play-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.DoraFlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.DoraFlatPPORunnerCfg,
    },
)
