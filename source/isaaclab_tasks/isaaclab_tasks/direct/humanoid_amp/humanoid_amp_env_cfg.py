# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING
from enum import Enum

from isaaclab_assets import HUMANOID_28_CFG

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .robot_registry import ROBOTS

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

class RobotTask(str, Enum):
    """Available robot and motion combinations."""
    DEFAULT_HUMANOID_WALK = "default_humanoid_walk"
    DEFAULT_HUMANOID_RUN = "default_humanoid_run"
    DEFAULT_HUMANOID_DANCE = "default_humanoid_dance"
    KBOT_WALK = "kbot_walk"

@configclass
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""
    
    # Robot selection - this determines everything else
    robot_task: RobotTask = RobotTask.DEFAULT_HUMANOID_WALK

    # env
    episode_length_s = 10.0
    decimation = 2

    # These will be set automatically based on robot_task in __post_init__
    observation_space: int = None
    action_space: int = None
    state_space: int = None
    num_amp_observations = 2
    amp_observation_space: int = None

    early_termination = True
    termination_height = 0.5

    # These will be set automatically based on robot_task
    motion_file: str = None
    reference_body: str = None
    key_body_names: list[str] = None

    reset_strategy = "random"  # default, random, random-start
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot - will be set automatically based on robot_task
    robot: ArticulationCfg = None

    def __post_init__(self):
        """Configure robot and spaces based on robot_task selection."""
        # Get the robot descriptor from registry
        if self.robot_task not in ROBOTS:
            raise ValueError(f"Unknown robot_task: {self.robot_task}. Available: {list(ROBOTS.keys())}")
        
        desc = ROBOTS[self.robot_task]
        
        # Set robot configuration
        self.robot = desc.cfg.replace(prim_path="/World/envs/env_.*/Robot").replace(
            actuators={
                "body": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    velocity_limit=100.0,
                    stiffness=None,
                    damping=None,
                ),
            },
        )
        
        # Set motion and body configuration
        self.motion_file = desc.motion_file
        self.reference_body = desc.reference_body
        self.key_body_names = desc.key_bodies
        
        # Set observation and action spaces
        robot_config = desc.robot_config
        self.observation_space = robot_config.observation_space
        self.action_space = robot_config.action_space
        self.state_space = robot_config.state_space
        self.amp_observation_space = robot_config.amp_observation_space


@configclass
class HumanoidAmpDanceEnvCfg(HumanoidAmpEnvCfg):
    robot_task: RobotTask = RobotTask.DEFAULT_HUMANOID_DANCE


@configclass
class HumanoidAmpRunEnvCfg(HumanoidAmpEnvCfg):
    robot_task: RobotTask = RobotTask.DEFAULT_HUMANOID_RUN


@configclass
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    robot_task: RobotTask = RobotTask.DEFAULT_HUMANOID_WALK


@configclass
class KBotAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    robot_task: RobotTask = RobotTask.KBOT_WALK
