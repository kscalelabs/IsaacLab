# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Stompy."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

DORA_CFG = ArticulationCfg(
    # Spawn stompy from URDF
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/dpsh/KIsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/dora/robot_fixed.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.8),
        joint_pos={
            "left_hip_pitch": 0.325,
            "left_hip_yaw": 0,
            "left_hip_roll": 0,
            "left_knee_pitch": -0.259,
            "left_ankle_pitch": -0.0556,
            "left_ankle_roll": 0,
            "right_hip_pitch": 0.325,
            "right_hip_yaw": 0,
            "right_hip_roll": 0,
            "right_knee_pitch": -0.259,
            "right_ankle_pitch": -0.0556,
            "right_ankle_roll": 0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*hip_yaw",
                ".*hip_roll",
                ".*hip_pitch",
                ".*knee_pitch",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*hip_yaw": 150.0,
                ".*hip_roll": 150.0,
                ".*hip_pitch": 200.0,
                ".*knee_pitch": 200.0,
            },
            damping={
                ".*hip_yaw": 5.0,
                ".*hip_roll": 5.0,
                ".*hip_pitch": 5.0,
                ".*knee_pitch": 5.0,
            },
            armature={
                ".*hip.*": 0.01,
                ".*knee.*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*ankle_pitch", ".*ankle_roll"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
