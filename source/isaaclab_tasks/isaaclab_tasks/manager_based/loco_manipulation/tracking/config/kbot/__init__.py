# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Tracking-LocoManip-KBot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
    "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:KBotLocoManipEnvCfg",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLocoManipPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Tracking-LocoManip-KBot-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
    "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:KBotLocoManipEnvCfg_PLAY",
    "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotLocoManipPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManip-KBot-RNN-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:KBotLocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:KBotFlatRNNPPORunnerCfg",
    },
)