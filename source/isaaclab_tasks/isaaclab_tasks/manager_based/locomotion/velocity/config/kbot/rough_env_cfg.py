"""Rough terrain locomotion environment config for kbot."""

import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
    ObservationTermCfg as ObsTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

from isaaclab_assets import KBOT_CFG


# Adds flat terrain to the terrain generator
# The default one did not have a flat portion
KBOT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0,
            grid_width=0.45,
            grid_height_range=(0.05, 0.2),
            platform_width=2.0,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.5,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


def velocity_push_curriculum(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    min_push: float,
    max_push: float,
    curriculum_start_step: int,
    curriculum_stop_step: int,
):
    """Progressively increases push velocity from min_push to max_push over specified steps.

    The return dict is logged to tensorboard.
    """
    # Only start curriculum after the specified start step
    if env.common_step_counter < curriculum_start_step:
        progress = 0.0
    else:
        # Calculate curriculum progress (0.0 to 1.0) from start_step to stop_step
        curriculum_duration = curriculum_stop_step - curriculum_start_step
        progress = (
            env.common_step_counter - curriculum_start_step
        ) / curriculum_duration
        progress = min(progress, 1.0)

    # Start with min velocity and increase to max
    current_velocity = min_push + (max_push - min_push) * progress

    # Update the push velocity range in the event manager
    if hasattr(env.event_manager.cfg, "push_robot"):
        env.event_manager.cfg.push_robot.params["velocity_range"] = {
            "x": (-current_velocity, current_velocity),
            "y": (-current_velocity, current_velocity),
        }

    # Return logging data for tensorboard
    return {
        "push_velocity_progress": progress,
        "push_velocity_magnitude": current_velocity,
    }


@configclass
class KBotRewards(RewardsCfg):
    """Reward terms for the K-Bot velocity task."""

    # -- base tracking & termination --
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"],
            ),
            "threshold": 0.4,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"],
            ),
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
            ),
        },
    )

    # Joint-limit & deviation penalties
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["dof_left_ankle_02", "dof_right_ankle_02"]
            )
        },
    )

    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "dof_left_hip_yaw_03",
                    "dof_right_hip_yaw_03",
                    "dof_left_hip_roll_03",
                    "dof_right_hip_roll_03",
                ],
            )
        },
    )

    joint_deviation_ankles = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["dof_left_ankle_02", "dof_right_ankle_02"]
            ),
        },
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    # left arm
                    "dof_left_shoulder_pitch_03",
                    "dof_left_shoulder_roll_03",
                    "dof_left_shoulder_yaw_02",
                    "dof_left_elbow_02",
                    "dof_left_wrist_00",
                    # right arm
                    "dof_right_shoulder_pitch_03",
                    "dof_right_shoulder_roll_03",
                    "dof_right_shoulder_yaw_02",
                    "dof_right_elbow_02",
                    "dof_right_wrist_00",
                ],
            )
        },
    )


@configclass
class KBotObservations:
    @configclass
    class CriticCfg(ObservationGroupCfg):
        # observation terms (order preserved)
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        # Replaced with privileged observations without noise below
        # joint_pos = ObsTerm(
        #     func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        # )
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )
        # IMU observations
        imu_projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        # Privileged Critic Observations
        # Joint dynamics information (privileged)
        joint_torques = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # Contact forces on feet (privileged foot contact information)
        feet_contact_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.01,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
                )
            },
        )

        # Body poses for important body parts (privileged state info)
        body_poses = ObsTerm(
            func=mdp.body_pose_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["base", "KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"],
                )
            },
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )

        # Joint positions and velocities with less noise (privileged accurate state)
        joint_pos_accurate = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),  # Much less noise than policy
        )
        joint_vel_accurate = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),  # Much less noise than policy
        )

        # Base position (full pose information - privileged)
        base_pos = ObsTerm(
            func=mdp.base_pos_z, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )

        # Root state information (privileged)
        root_lin_vel_w = ObsTerm(
            func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )
        root_ang_vel_w = ObsTerm(
            func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001)
        )

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # observation terms (order preserved)
        projected_gravity = ObsTerm(
            func=mdp.imu_projected_gravity,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # IMU observations
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        actions = ObsTerm(func=mdp.last_action)
        # No linear acceleration for now
        # imu_lin_acc = ObsTerm(
        #     func=mdp.imu_lin_acc,
        #     params={"asset_cfg": SceneEntityCfg("imu")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1)
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # Observation groups:
    critic: CriticCfg = CriticCfg()
    policy: PolicyCfg = PolicyCfg()


@configclass
class KBotCurriculumCfg:
    """Curriculum configuration for KBot push training."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # Configurable velocity push curriculum
    velocity_push_curriculum = CurrTerm(
        func=velocity_push_curriculum,
        params={
            "min_push": 0.01,
            "max_push": 2.0,
            "curriculum_start_step": 24 * 100,
            "curriculum_stop_step": 24 * 2500,
        },
    )


@configclass
class KBotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KBotRewards = KBotRewards()
    observations: KBotObservations = KBotObservations()
    curriculum: KBotCurriculumCfg = KBotCurriculumCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Scene
        self.scene.robot = KBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

        # Terrains
        # Override terrain generator with custom KBot configuration
        self.scene.terrain.terrain_generator = KBOT_ROUGH_TERRAINS_CFG

        # Enable curriculum for the custom terrain generator
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True

        # Imu
        self.scene.imu = ImuCfg(
            prim_path="{ENV_REGEX_NS}/Robot/imu",
            update_period=0.0,
            debug_vis=True,
            gravity_bias=(0.0, 0.0, 0.0),
        )

        # Physics material randomization (friction with the floor)
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
                ),
                "static_friction_range": (0.1, 2.0),
                "dynamic_friction_range": (0.1, 2.0),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,
                "make_consistent": True,  # Ensure dynamic friction is always less than static friction
            },
        )

        # # Base mass randomization BUG it does not seem to affect performance???
        # self.events.add_base_mass = EventTerm(
        #     func=mdp.randomize_rigid_body_mass,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names="Torso_Side_Right"),
        #         "mass_distribution_params": (0.001, 100.2),
        #         "operation": "scale",
        #         "distribution": "uniform",
        #         "recompute_inertia": True,
        #     },
        # )

        # Individual link mass randomization for robustness
        self.events.add_limb_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        # "Torso_Side_Right",
                        "KC_D_102L_L_Hip_Yoke_Drive",
                        "KC_C_104L_PitchHardstopDriven",
                        "KC_D_102R_R_Hip_Yoke_Drive",
                        "KC_C_104R_PitchHardstopDriven",
                        "RS03_5",
                        "RS03_6",
                        "RS03_4",
                        "RS03_3",
                        "KC_D_301L_L_Femur_Lower_Drive",
                        "KC_C_202L",
                        "KC_D_301R_R_Femur_Lower_Drive",
                        "KC_C_202R",
                        "KC_D_401L_L_Shin_Drive",
                        "KC_C_401L_L_UpForearmDrive",
                        "KC_D_401R_R_Shin_Drive",
                        "KC_C_401R_R_UpForearmDrive",
                        "KB_D_501L_L_LEG_FOOT",
                        "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop",
                        "KB_D_501R_R_LEG_FOOT",
                        "KB_C_501X_Right_Bayonet_Adapter_Hard_Stop",
                    ],
                ),
                "mass_distribution_params": (0.5, 1.5),  # Limb mass variations
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

        # # 3. CENTER OF MASS RANDOMIZATION BUG
        # self.events.base_com = EventTerm(
        #     func=mdp.randomize_rigid_body_com,
        #     mode="startup",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names="Torso_Side_Right"),
        #         "com_range": {
        #             "x": (-10.15, 10.03),   # metres
        #             "y": (-10.03, 10.03),
        #             "z": (-10.02, 10.02),
        #         },
        #     },
        # )

        # 4. ACTUATOR GAIN RANDOMIZATION
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.5, 1.5),  # ±50% stiffness variation
                "damping_distribution_params": (0.5, 1.5),  # ±50% damping variation
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # 5. JOINT PARAMETER RANDOMIZATION
        self.events.randomize_joint_properties = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.0, 1.5),  # Joint friction
                "armature_distribution_params": (0.4, 1.5),  # Joint armature/inertia
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # UNTESTED
        # # 6. JOINT INITIALIZATION RANDOMIZATION (Enable meaningful ranges)
        # self.events.reset_robot_joints.params["position_range"] = (-0.15, 0.15)  # ±15% around default
        # self.events.reset_robot_joints.params["velocity_range"] = (-1.0, 1.0)   # Small initial velocities

        # UNTESTED
        # # 7. EXTERNAL FORCE DISTURBANCES (Re-enable with random forces)
        # self.events.base_external_force_torque = EventTerm(
        #     func=mdp.apply_external_force_torque,
        #     mode="reset",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        #         "force_range": (-30.0, 30.0),   # Random forces in N
        #         "torque_range": (-15.0, 15.0),  # Random torques in Nm
        #     },
        # )

        self.events.push_robot.mode = "interval"
        self.events.push_robot.interval_range_s = (5.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
        }

        # Keep other existing randomization settings
        self.events.add_base_mass = None
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.base_com = None

        # UNTESTED
        # # 8. ENHANCED VELOCITY PUSHING
        # self.events.push_robot.mode = "interval"
        # self.events.push_robot.interval_range_s = (3.0, 8.0)  # More frequent pushes
        # self.events.push_robot.params["velocity_range"] = {
        #     "x": (-0.5, 0.5),  # Start with higher base velocity
        #     "y": (-0.3, 0.3),
        # }

        # UNTESTED
        # # 9. GRAVITY RANDOMIZATION
        # self.events.randomize_gravity = EventTerm(
        #     func=mdp.randomize_physics_scene_gravity,
        #     mode="startup",
        #     params={
        #         "gravity_distribution_params": ([-9.81, -9.81, -10.5], [-9.81, -9.81, -9.0]),  # Gravity variations
        #         "operation": "abs",
        #         "distribution": "uniform",
        #     },
        # )

        # UNTESTED
        # # 10. ENHANCED INITIAL POSE RANDOMIZATION
        # self.events.reset_base.params = {
        #     "pose_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8), "yaw": (-3.14, 3.14)},
        #     "velocity_range": {
        #         "x": (-0.3, 0.3),
        #         "y": (-0.3, 0.3),
        #         "z": (-0.1, 0.1),
        #         "roll": (-0.2, 0.2),
        #         "pitch": (-0.2, 0.2),
        #         "yaw": (-0.5, 0.5),
        #     },
        # }

        # UNTESTED
        # # 11. COLLIDER PROPERTY RANDOMIZATION
        # self.events.randomize_collider_properties = EventTerm(
        #     func=mdp.randomize_rigid_body_collider_offsets,
        #     mode="startup",
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
        #         "rest_offset_distribution_params": (-0.002, 0.002),     # Contact behavior variation
        #         "contact_offset_distribution_params": (0.001, 0.005),   # Contact detection variation
        #         "operation": "add",
        #         "distribution": "uniform",
        #     },
        # )

        # Rewards
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.dof_acc_l2.weight = -1.25e-7
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                # hips + knees only
                "dof_left_hip_pitch_04",
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_left_knee_04",
                "dof_right_hip_pitch_04",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
                "dof_right_knee_04",
            ],
        )
        self.rewards.dof_torques_l2.weight = -1.5e-7
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot",
            joint_names=[
                # hips + knees + ankles
                "dof_left_hip_pitch_04",
                "dof_left_hip_roll_03",
                "dof_left_hip_yaw_03",
                "dof_left_knee_04",
                "dof_left_ankle_02",
                "dof_right_hip_pitch_04",
                "dof_right_hip_roll_03",
                "dof_right_hip_yaw_03",
                "dof_right_knee_04",
                "dof_right_ankle_02",
            ],
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # Terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = [
            "base",
            "KC_D_102L_L_Hip_Yoke_Drive",
            "RS03_5",
            "KC_D_301L_L_Femur_Lower_Drive",
            "KC_D_401L_L_Shin_Drive",
            "KC_C_104L_PitchHardstopDriven",
            "RS03_6",
            "KC_C_202L",
            "KC_C_401L_L_UpForearmDrive",
            "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop",
            "KC_D_102R_R_Hip_Yoke_Drive",
            "RS03_4",
            "KC_D_301R_R_Femur_Lower_Drive",
            "KC_D_401R_R_Shin_Drive",
            "KC_C_104R_PitchHardstopDriven",
            "RS03_3",
            "KC_C_202R",
            "KC_C_401R_R_UpForearmDrive",
            "KB_C_501X_Right_Bayonet_Adapter_Hard_Stop",
        ]


@configclass
class KBotRoughEnvCfg_PLAY(KBotRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing for play
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Disable push curriculum for play mode
        self.curriculum.velocity_push_curriculum = None
