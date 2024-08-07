# Copyright (c) 2024 KScale Labs.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Stompy."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

STOMPYMINI_CFG = ArticulationCfg(
    # Spawn stompy from URDF
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/dpsh/KIsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/stompy_mini/robot_fixed.usd",
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
        pos=(0.0, 0.0, 1.00),
        rot=(0.7071, 0.7071, 0.0, 0.0),
        joint_pos={
            # left arm
            "left_shoulder_pitch": -1.02,
            "left_shoulder_yaw": 1.38,
            "left_shoulder_roll": -3.24,
            "left_elbow_pitch": 1.2,
            "left_wrist_roll": 0,
            # right arm
            "right_shoulder_pitch": 3.12,
            "right_shoulder_yaw": -1.98,
            "right_shoulder_roll": -1.38,
            "right_elbow_pitch": 1.32,
            "right_wrist_roll": 0,
            # legs
            "right_hip_pitch": 3.06,
            "right_hip_yaw": 3.24,
            "right_hip_roll": 3.18,
            "right_knee_pitch": 0,
            "right_ankle_pitch": 0.42,
            "left_hip_pitch": -0.06,
            "left_hip_yaw": 1.62,
            "left_hip_roll": 1.5,
            "left_knee_pitch": 0,
            "left_ankle_pitch": -1.62,
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
            joint_names_expr=[".*ankle_pitch"],
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
        "wrists": ImplicitActuatorCfg(
            joint_names_expr=[".*wrist_roll"],
            effort_limit=20,
            velocity_limit=100.0,
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
    },
)
