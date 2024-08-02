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
    spawn=sim_utils.UrdfFileCfg(
        asset_path="/home/dpsh/KIsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/locomotion/velocity/config/stompy/stompy/robot_fixed.urdf",
        fix_base=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.74),
        joint_pos={
            "torso roll": -0.502,
            # left arm
            "left shoulder pitch": -0.502,
            "left shoulder yaw": 1.26,
            "left shoulder roll": -3.01,
            "left elbow pitch": 4.46,
            "left wrist roll": -1.57,
            "left wrist pitch": 0,
            "left wrist yaw": 0.0628,
            # right arm
            "right shoulder pitch": 2.95,
            "right shoulder yaw": -1.26,
            "right shoulder roll": -0.126,
            "right elbow pitch": 1.13,
            "right wrist roll": -1.76,
            "right wrist pitch": 2.95,
            "right wrist yaw": 0.251,
            # legs
            "right hip pitch": -0.988,
            "right hip yaw": 1.07,
            "right hip roll": 0,
            "right knee pitch": 0.879,
            "right ankle pitch": 0.358,
            "right ankle roll": 1.76,
            "left hip pitch": 0.502,
            "left hip yaw": -2.07,
            "left hip roll": -1.57,
            "left knee pitch": 2.99,
            "left ankle pitch": 1,
            "left ankle roll": 1.76,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "*hip yaw",
                "*hip roll",
                "*hip pitch",
                "*knee pitch",
                "torso roll",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness={
                "*hip yaw": 150.0,
                "*hip roll": 150.0,
                "*hip pitch": 200.0,
                "*knee pitch": 200.0,
                "torso roll": 200.0,
            },
            damping={
                "*hip yaw": 5.0,
                "*hip roll": 5.0,
                "*hip pitch": 5.0,
                "*knee joint": 5.0,
                "torso roll": 5.0,
            },
            armature={
                "*hip*": 0.01,
                "*knee": 0.01,
                "torso roll": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=20,
            joint_names_expr=["*ankle pitch", "* ankle roll"],
            stiffness=20.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "*shoulder pitch",
                "*shoulder roll",
                "*shoulder yaw",
                "*elbow pitch",
                "*elbow roll",
            ],
            effort_limit=300,
            velocity_limit=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                "*shoulder*": 0.01,
                "*elbow*": 0.01,
            },
        ),
    },
)
