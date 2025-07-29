"""Rough terrain locomotion environment config for kbot."""

import math
from typing import Dict, Optional, Tuple
from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.terrains as terrain_gen
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
    CommandTerm,
    CommandTermCfg,
)
from isaaclab.sensors import ImuCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_unique



from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets import KBOT_CFG
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.markers import VisualizationMarkers

def quat_to_euler_xyz(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to Euler angles (roll, pitch, yaw).
    
    Args:
        quat: Quaternion tensor in (w, x, y, z) format. Shape: (..., 4)
        
    Returns:
        Euler angles tensor in (roll, pitch, yaw) format. Shape: (..., 3)
    """
    # Extract quaternion components
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    
    # Roll (x-axis rotation)
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    
    # Pitch (y-axis rotation)
    sin_pitch = 2 * (w * y - z * x)
    sin_pitch = torch.clamp(sin_pitch, -1.0, 1.0)  # Clamp to handle numerical errors
    pitch = torch.asin(sin_pitch)
    
    # Yaw (z-axis rotation)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    
    # Stack and return
    return torch.stack([roll, pitch, yaw], dim=-1)
@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for uniform height command generator.
    
    Uses height scanner position as the robot's current height and compares it
    against terrain height + absolute height command.
    """
    
    class_type: type = MISSING  # Will be set to UniformHeightCommand
    
    asset_name: str = MISSING
    """Name of the robot asset in the scene (used for reference only)."""
    
    scanner_cfg: SceneEntityCfg = MISSING
    """Configuration for the height scanner to measure terrain height and current position."""
    
    @configclass
    class Ranges:
        """Ranges for the height command."""
        height: tuple[float, float] = MISSING
        """Range for absolute height above terrain (in meters)."""
    
    ranges: Ranges = MISSING
    """Distribution ranges for absolute height commands above terrain."""


@configclass 
class UniformOrientationCommandCfg(CommandTermCfg):
    """Configuration for uniform orientation command generator.
    
    Generates Euler angle commands (roll, pitch, yaw) with a configurable 
    probability of issuing zero commands for neutral orientation practice.
    """
    
    class_type: type = MISSING  # Will be set to UniformOrientationCommand
    
    asset_name: str = MISSING
    """Name of the robot asset in the scene."""
    
    body_name: str = MISSING
    """Name of the body for which the command is generated."""
    
    zero_command_prob: float = 0.3
    """Probability of generating zero orientation command (neutral pose)."""
    
    @configclass
    class Ranges:
        """Ranges for the orientation command."""
        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in radians)."""
        pitch: tuple[float, float] = MISSING 
        """Range for the pitch angle (in radians)."""
        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in radians)."""
    
    ranges: Ranges = MISSING
    """Distribution ranges for the orientation commands."""


class UniformHeightCommand(CommandTerm):
    """Command generator for generating height commands uniformly.
    
    The command generator generates absolute height commands by sampling uniformly 
    within specified ranges for height above terrain.
    Uses the height scanner position as the robot's current height.
    """
    
    cfg: UniformHeightCommandCfg
    """Configuration for the command generator."""
    
    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.
        
        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        
        # extract height scanner (used for both terrain measurement and current position)
        self.height_scanner = env.scene.sensors[cfg.scanner_cfg.name]
        
        # create buffers
        # -- command: absolute height above terrain (1D)
        self.height_command = torch.zeros(self.num_envs, 1, device=self.device)
        # -- metrics
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)
    
    def __str__(self) -> str:
        msg = "UniformHeightCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeight range: {self.cfg.ranges.height}m above terrain\n"
        msg += f"\tUses height scanner position as current height\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        """The desired height command. Shape is (num_envs, 1)."""
        return self.height_command
    
    def _update_metrics(self):
        # Get current scanner height (this is our "robot height")
        current_height = self.height_scanner.data.pos_w[:, 2]
        
        terrain_height = torch.mean(self.height_scanner.data.ray_hits_w[..., 2], dim=-1)

        target_height = terrain_height + self.height_command[:, 0]

        # Compute height error
        self.metrics["height_error"] = torch.abs(current_height - target_height)
    
    def _resample_command(self, env_ids: Sequence[int]):
        # Sample new absolute height commands
        r = torch.empty(len(env_ids), device=self.device)
        self.height_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.height)
    
    def _update_command(self):
        pass

class UniformOrientationCommand(CommandTerm):
    """Command generator for generating orientation commands uniformly.
    
    The command generator generates orientation commands as Euler angles 
    (roll, pitch, yaw) by sampling uniformly within specified ranges.
    """
    
    cfg: UniformOrientationCommandCfg
    """Configuration for the command generator."""
    
    def __init__(self, cfg: UniformOrientationCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.
        
        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)
        
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        
        # create buffers
        # command: orientation as Euler angles (roll, pitch, yaw)
        self.orientation_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)
    
    def __str__(self) -> str:
        msg = "UniformOrientationCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tZero command probability: {self.cfg.zero_command_prob * 100:.1f}%\n"
        msg += f"\tOrientation ranges: roll={self.cfg.ranges.roll}, pitch={self.cfg.ranges.pitch}, yaw={self.cfg.ranges.yaw}\n"
        return msg
    
    @property
    def command(self) -> torch.Tensor:
        """The desired orientation command as Euler angles (roll, pitch, yaw). Shape is (num_envs, 3)."""
        return self.orientation_command
    
    def _update_metrics(self):
        # Get current body orientation as quaternion
        current_quat = self.robot.data.body_quat_w[:, self.body_idx]
        
        # Convert current quaternion to Euler angles
        current_euler = quat_to_euler_xyz(current_quat)  # (num_envs, 3)
        
        # Target Euler angles
        target_euler = self.orientation_command  # (num_envs, 3)
        
        euler_diff = torch.abs(current_euler - target_euler)
        euler_diff = torch.min(euler_diff, 2 * math.pi - euler_diff)
        
        # Sum of absolute angular errors
        self.metrics["orientation_error"] = torch.sum(euler_diff, dim=-1)
    
    def _resample_command(self, env_ids: Sequence[int]):
        # Determine which environments get zero commands vs sampled commands
        num_envs = len(env_ids)
        random_vals = torch.rand(num_envs, device=self.device)
        zero_mask = random_vals < self.cfg.zero_command_prob
        sample_mask = ~zero_mask
        
        # Set zero commands for selected environments
        if zero_mask.any():
            zero_env_indices = torch.tensor(env_ids, device=self.device)[zero_mask]
            self.orientation_command[zero_env_indices, :] = 0.0
        
        # Sample new orientation targets for remaining environments
        if sample_mask.any():
            sample_env_indices = torch.tensor(env_ids, device=self.device)[sample_mask]
            self.orientation_command[sample_env_indices, 0].uniform_(*self.cfg.ranges.roll)   # roll
            self.orientation_command[sample_env_indices, 1].uniform_(*self.cfg.ranges.pitch) # pitch
            self.orientation_command[sample_env_indices, 2].uniform_(*self.cfg.ranges.yaw)   # yaw
    
    def _update_command(self):
        pass


# Set the class type after the class is defined  
UniformOrientationCommandCfg.class_type = UniformOrientationCommand


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
    
    action_vel_current = current_action - prev_action_1
    action_vel_prev = prev_action_1 - prev_action_2
    action_acceleration = action_vel_current - action_vel_prev
    
    # Return L2 squared penalty
    return torch.sum(torch.square(action_acceleration), dim=1)


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


def foot_height_reward(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg, 
    contact_sensor_cfg: SceneEntityCfg,
    target_height: float, 
    std: float,
    ground_offset: float = 0.1
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.
        
    Args:
        env: The environment instance
        asset_cfg: Asset configuration for the robot with foot body names
        contact_sensor_cfg: Contact sensor configuration for detecting foot contact
        target_height: Target foot height above ground (in meters)
        std: Standard deviation for the exponential reward kernel
        ground_offset: Estimated offset from base to ground level (in meters)
        
    Returns:
        Reward tensor based on foot height during swing phase. Shape is (num_envs,).
    """
    # Extract the robot asset and contact sensor
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    
    # Get foot heights and base height (all vectorized)
    foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # (num_envs, num_feet)
    base_height = asset.data.root_pos_w[:, 2]  # (num_envs,)
    
    # Estimate ground level as base height minus offset
    ground_level = base_height - ground_offset  # (num_envs,)
    
    # Calculate relative foot height above ground
    relative_heights = foot_heights - ground_level.unsqueeze(1)  # (num_envs, num_feet)
    
    # Detect swing phase using contact forces (vectorized)
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, contact_sensor_cfg.body_ids, :]
    is_in_contact = torch.norm(contact_forces, dim=-1) > 10.0  # (num_envs, num_feet)
    is_in_swing = ~is_in_contact
    
    # Calculate height error only for swinging feet
    height_error = torch.square(relative_heights - target_height)  # (num_envs, num_feet)
    swing_gated_error = height_error * is_in_swing.float()
    
    # Sum error across feet and apply exponential reward
    total_error = torch.sum(swing_gated_error, dim=1)  # (num_envs,)
    reward = torch.exp(-total_error / (std**2))
    
    return reward

def track_height_command_reward(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    scanner_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward the robot for tracking the height command using scanner position.
    
    This function rewards the robot for maintaining the commanded absolute height
    above terrain. Uses the height scanner position as the current robot height.
    
    Args:
        env: The environment instance
        std: Standard deviation for the exponential reward kernel
        command_name: Name of the height command
        scanner_cfg: Configuration for the height scanner
        
    Returns:
        Reward tensor based on height tracking error. Shape is (num_envs,).
    """
    # Extract height scanner
    height_scanner = env.scene.sensors[scanner_cfg.name]
    
    current_height = height_scanner.data.pos_w[:, 2]  # (num_envs,)
    
    height_command = env.command_manager.get_command(command_name)  # (num_envs, 1)
    
    terrain_height = torch.mean(height_scanner.data.ray_hits_w[..., 2], dim=-1)
    target_height = terrain_height + height_command[:, 0]  # (num_envs,)
    
    height_error = torch.square(current_height - target_height)  # (num_envs,)
    
    reward = torch.exp(-height_error / (std**2))
    
    return reward


def track_body_orientation_axis_reward(
    env: ManagerBasedRLEnv, 
    std: float, 
    command_name: str, 
    asset_cfg: SceneEntityCfg,
    axis: str
) -> torch.Tensor:
    """Reward the robot for tracking a specific axis of body orientation command.
    
    This function rewards the robot for maintaining the commanded body orientation
    for a specific axis (roll, pitch, or yaw) allowing fine-grained control over
    orientation tracking rewards.
    
    Args:
        env: The environment instance
        std: Standard deviation for the exponential reward kernel
        command_name: Name of the orientation command
        asset_cfg: Asset configuration specifying which body to track
        axis: Which axis to track - "x", "y", or "z"
        
    Returns:
        Reward tensor based on single-axis orientation tracking error. Shape is (num_envs,).
    """
    # Map axis names to indices
    axis_map = {"x": 0, "y": 1, "z": 2}
    if axis not in axis_map:
        raise ValueError(f"Invalid axis '{axis}'. Must be one of: {list(axis_map.keys())}")
    
    axis_idx = axis_map[axis]
    
    # Extract the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get current body orientation (quaternion) and convert to Euler
    current_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]  # (num_envs, num_bodies, 4)
    current_euler = quat_to_euler_xyz(current_quat)  # (num_envs, num_bodies, 3)
    
    # Get the orientation command (Euler angles)
    target_euler_command = env.command_manager.get_command(command_name)  # (num_envs, 3)
    
    # Extract current and target angles for the specified axis
    current_angle = current_euler[:, :, axis_idx]  # (num_envs, num_bodies)
    target_angle = target_euler_command[:, axis_idx]  # (num_envs,)
    
    # Expand target to match body dimensions
    target_angle = target_angle.unsqueeze(1).expand(-1, current_angle.shape[1])  # (num_envs, num_bodies)
    
    # Calculate angular difference (handling wrap-around)
    angle_diff = torch.abs(current_angle - target_angle)  # (num_envs, num_bodies)
    angle_diff = torch.min(angle_diff, 2 * math.pi - angle_diff)  # Handle 2π wrap-around
    
    # Sum across all bodies for this axis
    total_axis_error = torch.sum(angle_diff, dim=1)  # (num_envs,)
    
    # Apply exponential reward kernel
    reward = torch.exp(-total_axis_error / (std**2))
    
    return reward

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
        weight=2.5,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
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

    foot_height = RewTerm(
        func=foot_height_reward,
        weight=0.8,
        params={
            "target_height": 0.15,  # 15cm target height for swing phase
            "std": 0.05,           # Standard deviation for exponential kernel
            "ground_offset": 0.1,   # Estimated offset from base to ground (10cm)
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"]
            ),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"],
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

    # Pose tracking rewards
    track_height_command = RewTerm(
        func=track_height_command_reward,
        weight=1.5,
        params={
            "std": 0.2,
            "command_name": "base_height",
            "scanner_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    track_body_orientation_x = RewTerm(
        func=track_body_orientation_axis_reward,
        weight=1.2,
        params={
            "std": 0.5,
            "command_name": "base_orientation", 
            "axis": "x",
            "asset_cfg": SceneEntityCfg("robot", body_names=["Torso_Side_Right"]),
        },
    )

    track_body_orientation_y = RewTerm(
        func=track_body_orientation_axis_reward,
        weight=1.2,
        params={
            "std": 0.5,
            "command_name": "base_orientation", 
            "axis": "y",
            "asset_cfg": SceneEntityCfg("robot", body_names=["Torso_Side_Right"]),
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
        height_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_height"}
        )
        orientation_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_orientation"}
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
        height_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_height"}
        )
        orientation_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_orientation"}
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
        self.scene.height_scanner.pattern_cfg.resolution = 0.08
        self.scene.height_scanner.pattern_cfg.size = [1.0, 1.0]

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
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.resampling_time_range = (0.5, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.2

        # Height and orientation commands
        pose_angles = math.radians(45)
        
        self.commands.base_height = UniformHeightCommandCfg(
            class_type=UniformHeightCommand,
            asset_name="robot",
            resampling_time_range=(4.0, 15.0),
            scanner_cfg=SceneEntityCfg("height_scanner"),
            ranges=UniformHeightCommandCfg.Ranges(
                height=(0.75, 1.05)  # 0.75m to 1.05m above terrain
            )
        )
        
        self.commands.base_orientation = UniformOrientationCommandCfg(
            class_type=UniformOrientationCommand,
            asset_name="robot", 
            body_name="Torso_Side_Right",
            resampling_time_range=(4.0, 15.0),
            zero_command_prob=0.4,
            ranges=UniformOrientationCommandCfg.Ranges(
                roll=(-pose_angles, pose_angles),
                pitch=(-pose_angles, pose_angles),
                yaw=(-pose_angles, pose_angles)
            )
        )

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
        self.scene.num_envs = 4
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

        self.commands.base_height.ranges.height = (1.05, 1.05)
        self.commands.base_orientation.zero_command_prob = 1.0
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing for play
        self.events.base_external_force_torque = None
        self.events.push_robot = None

        # Disable push curriculum for play mode
        self.curriculum.velocity_push_curriculum = None
        
        # Disable all domain randomization to save memory
        self.enable_randomization = False
