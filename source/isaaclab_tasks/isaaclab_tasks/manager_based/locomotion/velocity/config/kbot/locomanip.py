import math
from isaaclab_tasks.manager_based.locomotion.velocity.config.kbot.rough_rnn_env_cfg import KBotRoughEnvCfg, KBotRewards, KBotObservations
from isaaclab.utils import configclass

from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp

from isaaclab.assets import RigidObjectCfg
import numpy as np

LEFT_WRIST_NAME = "PRT0001_2"
RIGHT_WRIST_NAME = "PRT0001"

@configclass
class KBotLocoManipRewards(KBotRewards):
    joint_deviation_arms = None

    joint_vel_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw.*"])},
    )

    left_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_WRIST_NAME),
            "command_name": "left_ee_pose",
        },
    )

    left_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_WRIST_NAME),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_WRIST_NAME),
            "command_name": "left_ee_pose",
        },
    )

    right_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_WRIST_NAME),
            "command_name": "right_ee_pose",
        },
    )

    right_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_WRIST_NAME),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_WRIST_NAME),
            "command_name": "right_ee_pose",
        },
    )


@configclass
class KBotLocoManipObservations:
    @configclass
    class CriticCfg(KBotObservations.CriticCfg):
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
    @configclass
    class PolicyCfg(KBotObservations.PolicyCfg):
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()

@configclass
class KBotLocoManipCommands:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=LEFT_WRIST_NAME,
        resampling_time_range=(0.5, 1.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(0.05, 0.50),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(math.pi / 2.0 - 0.1, math.pi / 2.0 + 0.1),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=RIGHT_WRIST_NAME,
        resampling_time_range=(0.5, 1.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(-0.50, -0.05),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(-math.pi / 2.0 - 0.1, -math.pi / 2.0 + 0.1),
        ),
    )


@configclass
class KBotEvents(EventCfg):
    # Add an external force to simulate a payload being carried.
    left_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=LEFT_WRIST_NAME),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    right_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=RIGHT_WRIST_NAME),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )

@configclass
class KBotManipRoughEnvCfg(KBotRoughEnvCfg):
    rewards: KBotLocoManipRewards = KBotLocoManipRewards()
    observations: KBotLocoManipObservations = KBotLocoManipObservations()
    commands: KBotLocoManipCommands = KBotLocoManipCommands()
    def __post_init__(self):
        # post init of parent
        super().__post_init__()


@configclass 
class KBotManipRoughEnvCfgPlay(KBotManipRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 0
        self.episode_length_s = 600
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.5, 1.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing for play
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Disable push curriculum for play mode
        self.curriculum.velocity_push_curriculum = None

        # Disable domain randomization for play mode
        self.curriculum.domain_randomize_curriculum = None
        self.commands.base_velocity.debug_vis = False
        self.commands.left_ee_pose.debug_vis = False
        self.commands.right_ee_pose.debug_vis = False

        # self.scene.robot.init_state.pos =  np.array([0, 0, 0.6])
        # for k in self.scene.robot.init_state.joint_pos.keys():
        #     self.scene.robot.init_state.joint_pos[k] = 0.0