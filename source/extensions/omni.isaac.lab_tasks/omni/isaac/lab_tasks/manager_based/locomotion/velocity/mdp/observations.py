from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedEnv

"""
Actions
"""


def last_n_actions(env: ManagerBasedEnv, n: int) -> torch.Tensor:
    """The last n actions sent to the environment.

    """
    if n > env.action_manager.num_past_actions:
        raise ValueError(f"Requested {n} past actions, but the environment only has {env.action_manager.num_past_actions} past actions.")
    return env.action_manager.past_actions[:, :n, :].reshape(env.num_envs, n*env.action_manager.total_action_dim)
