"""Rough terrain locomotion environment config for kbot."""

import math
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
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import RigidObject, Articulation


def action_acceleration_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the acceleration (second derivative) of actions using L2 squared kernel.
    
    Requires action history length >= 2 to calculate acceleration.
    If insufficient history, returns zeros.
    """
    # Check if we have enough action history
    if env.action_manager.action_history_length < 2:
        return torch.zeros(env.num_envs, device=env.device)
    
    # Get action history: current, 1 step back, 2 steps back
    current_action = env.action_manager.action  # t
    prev_action_1 = env.action_manager.get_action_from_history(steps_back=0)  # t-1 (most recent in history)
    prev_action_2 = env.action_manager.get_action_from_history(steps_back=1)  # t-2
    
    # Calculate action velocity (first derivative)
    action_vel_current = current_action - prev_action_1  # v(t) = a(t) - a(t-1)
    action_vel_prev = prev_action_1 - prev_action_2      # v(t-1) = a(t-1) - a(t-2)
    
    # Calculate action acceleration (second derivative)  
    action_acceleration = action_vel_current - action_vel_prev  # acc(t) = v(t) - v(t-1)
    
    # Return L2 squared penalty
    return torch.sum(torch.square(action_acceleration), dim=1)

def body_distance_penalty(
    env: ManagerBasedRLEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg,
    body_a_names: list[str],
    body_b_names: list[str],
) -> torch.Tensor:
    """Penalize when specific bodies get too close to each other.
    
    This function is useful for preventing collisions between specific body parts, 
    such as foot-to-foot collisions or arm-leg collisions.
    
    Args:
        env: The environment instance.
        min_distance: The minimum allowed distance between bodies.
        asset_cfg: The asset configuration containing the bodies.
        body_a_names: List of names for the first set of bodies.
        body_b_names: List of names for the second set of bodies.
        
    Returns:
        A penalty tensor based on proximity violations. Shape is (num_envs,).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Get body indices for the specified body names
    body_a_ids, _ = asset.find_bodies(body_a_names)
    body_b_ids, _ = asset.find_bodies(body_b_names)
    
    # Get body positions in world frame
    body_a_pos = asset.data.body_pos_w[:, body_a_ids]  # (num_envs, num_bodies_a, 3)
    body_b_pos = asset.data.body_pos_w[:, body_b_ids]  # (num_envs, num_bodies_b, 3)
    
    # Compute pairwise distances between all body_a and body_b pairs
    penalty = torch.zeros(env.num_envs, device=env.device)
    
    for i in range(len(body_a_ids)):
        for j in range(len(body_b_ids)):
            # Calculate distance between body_a[i] and body_b[j]
            distance = torch.norm(body_a_pos[:, i] - body_b_pos[:, j], dim=1)
            # Penalize when distance is below minimum
            violation = torch.clamp(min_distance - distance, min=0.0)
            penalty += violation
    
    return penalty

_AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}


def body_distance_axis_penalty(
    env: ManagerBasedRLEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg,
    body_a_names: list[str],
    body_b_names: list[str],
    axis: str = "y",
) -> torch.Tensor:
    """Penalize bodies that get too close *per Cartesian axis*.

    Instead of the Euclidean distance, this penalty is applied separately on
    each axis.  For a body pair, if the absolute separation along the *x*, *y*,
    or *z* axis falls below ``min_distance``, the violation amount
    ``(min_distance - |delta|)`` is accumulated.  This is useful when you want
    to discourage overlapping in individual directions (e.g. legs crossing in
    the *y* direction) without being overly restrictive on diagonal motion.
    """

    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve body indices once
    body_a_ids, _ = asset.find_bodies(body_a_names)
    body_b_ids, _ = asset.find_bodies(body_b_names)

    # Positions
    pos_a = asset.data.body_pos_w[:, body_a_ids]  # (N, A, 3)
    pos_b = asset.data.body_pos_w[:, body_b_ids]  # (N, B, 3)

    penalty = torch.zeros(env.num_envs, device=env.device)

    # Validate axis
    if axis not in _AXIS_TO_IDX:
        raise ValueError(f"axis must be one of {_AXIS_TO_IDX.keys()}, got '{axis}'")

    idx = _AXIS_TO_IDX[axis]

    for i in range(len(body_a_ids)):
        for j in range(len(body_b_ids)):
            delta = torch.abs(pos_a[:, i, idx] - pos_b[:, j, idx])  # (N,)
            violation = torch.clamp(min_distance - delta, min=0.0)  # (N,)
            penalty += violation

    return penalty


def contact_forces_l2_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize contact forces using L2 squared penalty above threshold.
    
    Computes L2 norm of contact force vector, then applies squared penalty for violations above threshold.
    """
    # Extract contact sensor data
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    
    current_force_magnitudes = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids], dim=-1)
    
    # Compute violation above threshold
    violations = torch.clamp(current_force_magnitudes - threshold, min=0.0)
    
    l2_penalty = torch.sum(torch.square(violations), dim=1) 
    return l2_penalty

def flat_orientation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # L1 penalty: sum of absolute xy-components of projected gravity.
    return torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1)


def foot_height_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    height_cap: float,
) -> torch.Tensor:
    """Reward feet for being lifted, saturating at ``height_cap``.

    The reward grows linearly with the z-height of each specified foot up to
    ``height_cap`` meters, after which it is capped.
    """

    asset = env.scene[asset_cfg.name]

    # Height of each foot in world frame (z-axis)
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    clipped = torch.clamp(foot_heights, max=height_cap)

    reward = torch.sum(clipped / height_cap, dim=1)

    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1

    return reward

def foot_height_swing_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    target_height: float, 
    std: float, 
    tanh_mult: float = 2.0
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.
    
    This function encourages the robot to lift its feet to a target height during swing phase.
    The reward is only applied when the foot is moving (velocity-gated), ensuring it only 
    rewards during swing and not stance phases.
    
    Args:
        env: The environment instance
        asset_cfg: Asset configuration for the robot with foot body names
        target_height: Target foot height above ground (in meters)
        std: Standard deviation for the exponential reward kernel
        tanh_mult: Multiplier for velocity tanh to determine swing phase
        
    Returns:
        Reward tensor based on foot height during swing phase. Shape is (num_envs,).
    """
    # Extract the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get foot positions (z-coordinate is height)
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    
    # Calculate height error from target
    height_error = torch.square(foot_heights - target_height)
    
    # Get foot horizontal velocity to determine if foot is in swing phase
    foot_velocity_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    foot_speed = torch.norm(foot_velocity_xy, dim=-1)
    
    # Use tanh to create a smooth velocity gate (0 when stationary, 1 when moving fast)
    velocity_gate = torch.tanh(tanh_mult * foot_speed)
    
    # Apply velocity gate to height error (only reward when foot is moving)
    gated_error = height_error * velocity_gate
    
    # Sum across all feet and apply exponential kernel
    total_error = torch.sum(gated_error, dim=1)
    reward = torch.exp(-total_error / std)
    
    return reward


def standing_lin_vel_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.1,
) -> torch.Tensor:
    """Penalize base linear velocity when the command is (almost) zero.

    This term encourages the robot to remain stable during *standing* episodes
    (i.e. when the commanded linear velocity magnitude is below ``threshold``).

    The returned value is the L1 norm of the xy-components of the base linear
    velocity expressed in the robot frame, gated so that it contributes **only**
    for standing commands.
    """

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    # commanded linear velocity (body frame)
    cmd_lin_vel_xy = env.command_manager.get_command(command_name)[:, :2]
    cmd_speed = torch.norm(cmd_lin_vel_xy, dim=1)

    # determine which envs are in standing mode
    is_standing = cmd_speed < threshold

    # L1 error (command is ~0) – use absolute actual velocity
    lin_vel_error = torch.sum(torch.abs(asset.data.root_lin_vel_b[:, :2]), dim=1)

    # only penalize standing envs
    return lin_vel_error * is_standing.float()

def foot_flat_orientation_l1(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    roll_offset: float = 0.0,
    pitch_offset: float = 0.0,
    yaw_offset: float = 0.0,
) -> torch.Tensor:
    """L1 penalty for the feet on roll/pitch: sum(|gravity_body_frame.xy|). Zero when flat.
    
    Note:
        The foot frame is wrong, so we need to rotate it by the given offset
        TODO: Root cause and correct in Onshape 
    """

    asset = env.scene[asset_cfg.name]

    # Get foot quat
    foot_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids]  # Shape: (num_envs, num_feet, 4)

    def correct_foot_quat(foot_quat_w: torch.Tensor, roll_offset: float, pitch_offset: float, yaw_offset: float) -> torch.Tensor:
        offset_quat = quat_from_euler_xyz(
            torch.full_like(foot_quat_w[:, :, 0], roll_offset),
            torch.full_like(foot_quat_w[:, :, 0], pitch_offset), 
            torch.full_like(foot_quat_w[:, :, 0], yaw_offset)
        )  # Shape: (num_envs, num_feet, 4)
        return math_utils.quat_mul(foot_quat_w, offset_quat)


    # By default the foot frame is wrong, so we need to rotate it by the given offset
    # TODO: Root cause and correct in Onshape 
    if roll_offset != 0.0 or pitch_offset != 0.0 or yaw_offset != 0.0:
        foot_quat_w = correct_foot_quat(foot_quat_w, roll_offset, pitch_offset, yaw_offset)

    # Get gravity vector in world frame
    gravity_dir = asset.data.GRAVITY_VEC_W.unsqueeze(1)  # Shape: (num_envs, 1, 3)

    # Expand gravity vector to match number of feet
    num_feet = len(asset_cfg.body_ids)
    gravity_dir = gravity_dir.expand(-1, num_feet, -1)  # Shape: (num_envs, num_feet, 3)

    # Project gravity onto each foot's local frame
    projected_gravity = math_utils.quat_apply_inverse(foot_quat_w, gravity_dir)  # Shape: (num_envs, num_feet, 3)

    # Penalize only the xy-components (roll and pitch)
    roll_pitch_error = projected_gravity[:, :, :2]  # Shape: (num_envs, num_feet, 2)

    # Compute L1 penalty for roll and pitch across all feet (sum across all feet)
    penalty = torch.sum(torch.abs(roll_pitch_error), dim=-1)  # Shape: (num_envs, num_feet)
    penalty = torch.sum(penalty, dim=-1)  # Sum across all feet, Shape: (num_envs,)
    return penalty

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
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.2},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # Standing stability penalty (active only for near-zero velocity commands)
    standing_lin_vel_l1 = RewTerm(
        func=standing_lin_vel_l1,
        weight=-1.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
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
            "threshold": 0.8,
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

    foot_height = RewTerm(
        func=foot_height_reward,
        weight=0.3,
        params={
            "height_cap": 0.2,  # cap reward growth at 20 cm
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
            ),
            "command_name": "base_velocity",
        },
    )

    # foot_height = RewTerm(
    #     func=foot_height_swing_reward,
    #     weight=0.5,
    #     params={
    #         "target_height": 0.2,  # 15cm target height for swing phase
    #         "std": 0.05,           # Standard deviation for exponential kernel
    #         "tanh_mult": 2.0,      # Velocity multiplier for swing detection
    #         "asset_cfg": SceneEntityCfg(
    #             "robot", body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
    #         ),
    #     },
    # )

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
                    # "dof_left_shoulder_pitch_03",
                    "dof_left_shoulder_roll_03",
                    "dof_left_shoulder_yaw_02",
                    "dof_left_elbow_02",
                    "dof_left_wrist_00",
                    # right arm
                    # "dof_right_shoulder_pitch_03",
                    "dof_right_shoulder_roll_03",
                    "dof_right_shoulder_yaw_02",
                    "dof_right_elbow_02",
                    "dof_right_wrist_00",
                ],
            )
        },
    )

    joint_deviation_shoulder_pitch = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.4,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    "dof_left_shoulder_pitch_03",
                    "dof_right_shoulder_pitch_03",
                ],
            )
        },
    )

    # Action smoothness penalty
    action_acceleration_l2 = RewTerm(
        func=action_acceleration_l2,
        weight=-0.1,
    )

    # Foot contact force penalty - L2 penalty above threshold
    foot_contact_force_l2 = RewTerm(
        func=contact_forces_l2_penalty,
        weight=-1.0e-7,
        params={
            "threshold": 360.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"],
            ),
        },
    )

    flat_orientation_l1 = RewTerm(
        func=flat_orientation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    feet_distance_penalty = RewTerm(
        func=body_distance_axis_penalty,
        weight=-0.1,
        params={
            "min_distance": 0.25,
            "asset_cfg": SceneEntityCfg("robot"),
            "body_a_names": ["KB_D_501L_L_LEG_FOOT"],
            "body_b_names": ["KB_D_501R_R_LEG_FOOT"],
            "axis": "x",
        },
    )

    foot_flat_orientation = RewTerm(
        func=foot_flat_orientation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[
                    "KB_D_501L_L_LEG_FOOT", 
                    "KB_D_501R_R_LEG_FOOT"
                    ]
            ),
            # Manually checked that roll offset is needed
            "roll_offset": math.radians(90),
            "pitch_offset": math.radians(0),
            "yaw_offset": math.radians(0),
        },
    )

    # No stomping reward
    # Foot-impact regulariser (discourages stomping)
    # foot_impact_penalty = RewTerm(
    #     func=mdp.contact_forces,
    #     weight=-1.5e-3,
    #     params={
    #         "threshold": 358.0,  # Manually checked static load of the kbot while standing
    #         "sensor_cfg": SceneEntityCfg(
    #             "contact_forces",
    #             body_names=[
    #                 "KB_D_501L_L_LEG_FOOT",
    #                 "KB_D_501R_R_LEG_FOOT",
    #                 "Torso_Side_Right",
    #                 "KC_D_102L_L_Hip_Yoke_Drive",
    #                 "RS03_5",
    #                 "KC_D_301L_L_Femur_Lower_Drive",
    #                 "KC_D_401L_L_Shin_Drive",
    #                 "KC_C_104L_PitchHardstopDriven",
    #                 "RS03_6",
    #                 "KC_C_202L",
    #                 "KC_C_401L_L_UpForearmDrive",
    #                 "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop",
    #                 "KC_D_102R_R_Hip_Yoke_Drive",
    #                 "RS03_4",
    #                 "KC_D_301R_R_Femur_Lower_Drive",
    #                 "KC_D_401R_R_Shin_Drive",
    #                 "KC_C_104R_PitchHardstopDriven",
    #                 "RS03_3",
    #                 "KC_C_202R",
    #                 "KC_C_401R_R_UpForearmDrive",
    #                 "KB_C_501X_Right_Bayonet_Adapter_Hard_Stop",
    #             ],
    #         ),
    #     },
    # )


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
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="Torso_Side_Right")

        # Rewards
        self.rewards.lin_vel_z_l2.weight = -1.0
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

        self.rewards.ankle_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.5e-6,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["dof_left_ankle_02", "dof_right_ankle_02"]),
            },
        )

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.3

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

        # Remove actuator latency
        for act_cfg in self.scene.robot.actuators.values():
            if hasattr(act_cfg, "min_delay"):
                act_cfg.min_delay = 0
            if hasattr(act_cfg, "max_delay"):
                act_cfg.max_delay = 0


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
