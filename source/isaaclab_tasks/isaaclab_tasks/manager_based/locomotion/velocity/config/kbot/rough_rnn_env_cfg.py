"""Rough terrain locomotion environment config for kbot."""

from typing import Dict, Optional, Tuple

import torch

import isaaclab.terrains as terrain_gen
import isaaclab.utils.math as math_utils
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg,
    ObservationTermCfg,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ImuCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets import KBOT_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)


def randomize_imu_mount(
    env: ManagerBasedEnv,
    env_ids: Optional[torch.Tensor],
    sensor_cfg: SceneEntityCfg,
    pos_range: Dict[str, Tuple[float, float]],
    rot_range: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Helper to randomise the IMU's local pose on every env reset."""
    imu_sensor = env.scene.sensors[sensor_cfg.name]

    # Get the envs which reset
    env_indices = (
        env_ids
        if env_ids is not None
        else torch.arange(imu_sensor.num_instances, device=env.device)
    )
    num_envs_to_update = len(env_indices)

    def sample_uniform(lo: float, hi: float) -> torch.Tensor:
        """Return `num_envs_to_update` samples from [lo, hi)."""
        return (hi - lo) * torch.rand(num_envs_to_update, device=env.device) + lo

    # Sample translation offsets
    position_offsets: torch.Tensor = torch.stack(
        [
            sample_uniform(*pos_range["x"]),
            sample_uniform(*pos_range["y"]),
            sample_uniform(*pos_range["z"]),
        ],
        dim=-1,  # shape = (N, 3)
    )

    # Sample orientation offsets
    roll_offsets = sample_uniform(*rot_range["roll"])
    pitch_offsets = sample_uniform(*rot_range["pitch"])
    yaw_offsets = sample_uniform(*rot_range["yaw"])

    quaternion_offsets: torch.Tensor = quat_from_euler_xyz(
        roll_offsets, pitch_offsets, yaw_offsets  # shape = (N, 4)
    )

    # Write the offsets into the sensor’s internal buffers
    imu_sensor._offset_pos_b[env_indices] = position_offsets
    imu_sensor._offset_quat_b[env_indices] = quaternion_offsets

    # Return summary scalars for logging / curriculum
    # Not sure if this is needed
    mean_offset_cm: float = (position_offsets.norm(dim=-1).mean() * 100.0).item()
    mean_tilt_deg: float = (
        torch.rad2deg(torch.acos(quaternion_offsets[:, 0].clamp(-1.0, 1.0)))
        .mean()
        .item()
    )

    return {
        "imu_offset_cm": mean_offset_cm,
        "imu_tilt_deg": mean_tilt_deg,
    }


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

    joint_deviation_hip_pitch_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "dof_left_hip_pitch_04",
                    "dof_right_hip_pitch_04",
                    "dof_left_knee_04",
                    "dof_right_knee_04",
                ],
            ),
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

    # No stomping reward
    # Foot-impact regulariser (discourages stomping)
    foot_impact_penalty = RewTerm(
        func=mdp.contact_forces,
        weight=-1.5e-3,
        params={
            "threshold": 358.0,  # Manually checked static load of the kbot while standing
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "KB_D_501L_L_LEG_FOOT",
                    "KB_D_501R_R_LEG_FOOT",
                    "Torso_Side_Right",
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
                ],
            ),
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
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
        )
        joint_vel_accurate = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.0001, n_max=0.0001),
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

        # No noise for the critic
        def __post_init__(self):
            self.enable_corruption = False

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
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        # IMU observations
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
        # No past actions for rnn
        # actions = ObsTerm(func=mdp.last_action)

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
            "max_push": 0.5,
            "curriculum_start_step": 24 * 500,
            "curriculum_stop_step": 24 * 5500,
        },
    )


@configclass
class KBotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    enable_randomization: bool = True
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
            offset=ImuCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)  # meters, quaternion
            ),
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

        # Individual link mass randomization for robustness
        self.events.add_limb_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=[
                        "Torso_Side_Right",
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
                "mass_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
                "recompute_inertia": True,
            },
        )

        # PD gains randomization
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stiffness_distribution_params": (0.8, 1.2),
                "damping_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # Actuator friction and armature randomization
        self.events.randomize_joint_properties = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "friction_distribution_params": (0.0, 0.3),
                "armature_distribution_params": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
            },
        )

        # Joint initialization randomization
        # Reset by offset is needed since the default is to scale by zero
        self.events.reset_robot_joints.params["position_range"] = (-0.2, 0.2)
        self.events.reset_robot_joints.params["velocity_range"] = (-1.0, 1.0)
        self.events.reset_robot_joints.func = mdp.reset_joints_by_offset

        self.events.push_robot.mode = "interval"
        self.events.push_robot.interval_range_s = (5.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
        }

        # Base reset randomization
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        }

        # IMU offset pos and rot randomization
        self.events.randomize_imu_mount = EventTerm(
            func=randomize_imu_mount,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
                "pos_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (-0.05, 0.05),
                },
                "rot_range": {
                    "roll": (-0.1, 0.1),
                    "pitch": (-0.1, 0.1),
                    "yaw": (-0.1, 0.1),
                },
            },
        )

        # I think this is because the "base" is not a rigid body in the robot asset
        self.events.add_base_mass = None
        self.events.base_com = None

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
        self.commands.base_velocity.rel_standing_envs = 0.2

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

        # Apply randomization settings based on flag
        if not self.enable_randomization:
            self._disable_randomization()

    def _disable_randomization(self):
        """Disable all randomization for easy early training.

        Use with command line arg: env.enable_randomization=false
        """
        
        print("[INFO]: Disabling all domain randomization!\n" * 5, end="")

        # Disable events
        self.events.physics_material = None
        self.events.add_limb_masses = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_properties = None
        self.events.randomize_imu_mount = None

        # Simple resets
        self.events.reset_robot_joints.params.update(
            {"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)}
        )
        self.events.reset_robot_joints.func = mdp.reset_joints_by_scale
        self.events.reset_base.params = {
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        # No pushes and push curriculum
        self.events.push_robot = None
        if hasattr(self.curriculum, "velocity_push_curriculum"):
            self.curriculum.velocity_push_curriculum = None

        # No actor observation noise
        self.observations.policy.enable_corruption = False

        # No foot impact penalty
        if hasattr(self.rewards, "foot_impact_penalty"):
            self.rewards.foot_impact_penalty = None


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
