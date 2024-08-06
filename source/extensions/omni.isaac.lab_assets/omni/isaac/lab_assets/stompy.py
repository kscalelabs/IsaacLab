# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Stompy."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

STOMPY_CFG = ArticulationCfg(
    # Spawn stompy from URDF
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/dpsh/KIsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/stompy/stompy/robot_fixed/robot_fixed.usd",
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
        pos=(0.0, 0.0, 1.25),
        joint_pos={
            "torso_roll": -0.502,
            # left arm
            "left_shoulder_pitch": -0.251,  # Corrected from -0.502
            "left_shoulder_yaw": 1.820,  # Corrected from 1.26
            "left_shoulder_roll": -1.445,  # Corrected from -3.01
            "left_elbow_pitch": 2.065,  # Corrected from 4.46
            "left_wrist_roll": -2.510,  # Corrected from -1.57
            "left_wrist_pitch": 3.325,  # Corrected from 0
            "left_wrist_yaw": 0.0628,  # No correction needed
            # right arm
            "right_shoulder_pitch": 2.700,  # Corrected from 2.95
            "right_shoulder_yaw": -1.820,  # Corrected from -1.26
            "right_shoulder_roll": -2.575,  # Corrected from -0.126
            "right_elbow_pitch": -2.575,  # Corrected from 1.13
            "right_wrist_roll": -0.005,  # Corrected from -1.76
            "right_wrist_pitch": 0.251,  # Corrected from 2.95
            "right_wrist_yaw": 1.375,  # Corrected from 0.251
            # legs
            "right_hip_pitch": 1.130,  # Corrected from -0.988
            "right_hip_yaw": 1.07,
            "right_hip_roll": 0,
            "right_knee_pitch": 0.879,
            "right_ankle_pitch": 0.358,
            "right_ankle_roll": 1.76,
            "left_hip_pitch": 0.502,
            "left_hip_yaw": -2.07,
            "left_hip_roll": -1.57,
            "left_knee_pitch": 2.99,
            "left_ankle_pitch": 1,
            "left_ankle_roll": 1.76,
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
                "torso_roll",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                ".*hip_yaw": 150.0,
                ".*hip_roll": 150.0,
                ".*hip_pitch": 200.0,
                ".*knee_pitch": 200.0,
                "torso_roll": 200.0,
            },
            damping={
                ".*hip_yaw": 5.0,
                ".*hip_roll": 5.0,
                ".*hip_pitch": 5.0,
                ".*knee_pitch": 5.0,
                "torso_roll": 5.0,
            },
            armature={
                ".*hip.*": 0.01,
                ".*knee.*": 0.01,
                "torso_roll": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=[".*ankle_pitch", ".*ankle_roll"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*shoulder_pitch",
                ".*shoulder_roll",
                ".*shoulder_yaw",
                ".*elbow_pitch",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*shoulder.*": 0.01,
                ".*elbow.*": 0.01,
            },
        ),
    },
)
