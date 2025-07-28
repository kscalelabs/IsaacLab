"""Rough terrain locomotion environment config for kbot."""

import math
from typing import Dict, Optional, Tuple
import torch

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
    ObservationTermCfg as ObsTerm,
    EventTermCfg as EventTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
import isaaclab.terrains as terrain_gen
from isaaclab.utils.math import quat_from_euler_xyz


import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    CurriculumCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

from isaaclab.envs import ManagerBasedRLEnv


from isaaclab_assets import KBOT_CFG

def latency_randomized_observation(
    env: ManagerBasedEnv,
    base_func: callable,
    base_func_kwargs: dict,
    latency_steps_range: tuple[int, int] = (1, 5),
    buffer_size: int = 10,
) -> torch.Tensor:
    """Observation function with randomized latency simulation.
    
    Optimized version with vectorized operations and minimal overhead.
    """
    min_latency, max_latency = latency_steps_range
    
    # Fast path: if no latency possible, just return current observation
    if max_latency == 0:
        return base_func(env, **base_func_kwargs)
    
    # Initialize buffer attribute name
    buffer_attr = f"_latency_buffer_{base_func.__name__}_{id(base_func_kwargs)}"
    step_attr = f"_latency_step_{base_func.__name__}_{id(base_func_kwargs)}"
    
    # Get current observation
    current_obs = base_func(env, **base_func_kwargs)
    
    # Initialize buffer if it doesn't exist
    if not hasattr(env, buffer_attr):
        setattr(env, buffer_attr, torch.zeros(
            (buffer_size, *current_obs.shape), 
            device=env.device, 
            dtype=current_obs.dtype
        ))
        setattr(env, step_attr, 0)
        # Return current observation on first call
        return current_obs
    
    # Get buffer and current step
    buffer = getattr(env, buffer_attr)
    current_step = getattr(env, step_attr)
    
    # Store current observation in buffer
    buffer[current_step % buffer_size] = current_obs
    
    # Update step counter
    setattr(env, step_attr, current_step + 1)
    
    # Check if we have enough history
    available_steps = min(current_step, buffer_size)
    max_available_latency = min(max_latency, available_steps - 1)
    
    if max_available_latency < min_latency:
        return current_obs
    
    # Optimized for (0,1) case - much faster
    if min_latency == 0 and max_latency == 1:
        # Random boolean mask for each environment
        use_delayed = torch.rand(env.num_envs, device=env.device) > 0.5
        
        if available_steps >= 1:
            prev_obs = buffer[(current_step - 1) % buffer_size]
            return torch.where(use_delayed.unsqueeze(-1), prev_obs, current_obs)
        else:
            return current_obs
    
    # General case for larger latency ranges
    latency_steps = torch.randint(
        min_latency, 
        max_available_latency + 1, 
        (env.num_envs,), 
        device=env.device
    )
    
    # Vectorized buffer indexing
    buffer_indices = (current_step - latency_steps) % buffer_size
    env_indices = torch.arange(env.num_envs, device=env.device)
    
    return buffer[buffer_indices, env_indices]

def imu_projected_gravity_with_latency(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("imu"),
    latency_steps_range: tuple[int, int] = (1, 5),
    buffer_size: int = 10,
) -> torch.Tensor:
    """IMU projected gravity with simulated latency.
    
    Args:
        env: The environment instance  
        asset_cfg: IMU sensor configuration
        latency_steps_range: (min_steps, max_steps) for latency simulation
                           - (0, 0): No latency (fastest)
                           - (0, 1): Binary latency (very fast)
                           - (1, 5): Multi-step latency (optimized)
        buffer_size: History buffer size (should be >= max_latency)
    
    Returns:
        Projected gravity with simulated latency
    """
    return latency_randomized_observation(
        env=env,
        base_func=mdp.imu_projected_gravity,
        base_func_kwargs={"asset_cfg": asset_cfg},
        latency_steps_range=latency_steps_range,
        buffer_size=buffer_size,
    )

def override_value(env, env_ids, data, value, num_steps):
    if env.common_step_counter > num_steps:
        return value
    return mdp.modify_term_cfg.NO_CHANGE

def unified_command_curriculum(
    env,
    env_ids,
    initial_ranges: dict[str, tuple[float, float]],
    target_ranges: dict[str, tuple[float, float]],
    start_step: int,
    end_step: int,
):
    """Unified curriculum for all command ranges with configurable targets.
    
    Args:
        initial_ranges: Dict mapping command names to initial (min, max) ranges
                       e.g., {"lin_vel_x": (-0.25, 0.25), "lin_vel_y": (-0.25, 0.25), "ang_vel_z": (-0.1, 0.1)}
        target_ranges: Dict mapping command names to target (min, max) ranges
                      e.g., {"lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.5, 1.5), "ang_vel_z": (-1.0, 1.0)}
    
    The return dict is logged to tensorboard.
    """
    if env.common_step_counter < start_step:
        progress = 0.0
    elif env.common_step_counter >= end_step:
        progress = 1.0
    else:
        # Linear interpolation between start and end steps
        progress = (env.common_step_counter - start_step) / (end_step - start_step)
    
    # Collect logging data
    logging_data = {
        "command_curriculum_progress": progress,
    }
    
    # Update all command ranges
    for command_name in initial_ranges.keys():
        if command_name not in target_ranges:
            continue  # Skip if no target specified
            
        initial_range = initial_ranges[command_name]
        target_range = target_ranges[command_name]
        
        if progress == 0.0:
            current_range = initial_range
        elif progress == 1.0:
            current_range = target_range
        else:
            # Linear interpolation
            min_val = initial_range[0] + progress * (target_range[0] - initial_range[0])
            max_val = initial_range[1] + progress * (target_range[1] - initial_range[1])
            current_range = (min_val, max_val)
        
        # Update the command range directly
        try:
            # Navigate to the attribute and set it (assumes base_velocity command manager)
            command_attr = f"command_manager.cfg.base_velocity.ranges.{command_name}"
            parts = command_attr.split('.')
            obj = env
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], current_range)
        except AttributeError:
            pass  # Silently fail if attribute doesn't exist
        
        # Add per-command logging
        logging_data.update({
            f"command_{command_name}_min": current_range[0],
            f"command_{command_name}_max": current_range[1],
            # f"command_{command_name}_span": current_range[1] - current_range[0],
            # f"command_{command_name}_target_min": target_range[0],
            # f"command_{command_name}_target_max": target_range[1],
            # f"command_{command_name}_target_span": target_range[1] - target_range[0],
        })
    
    return logging_data


def unified_push_curriculum(
    env,
    env_ids,
    target_velocities: dict[str, tuple[float, float]],
    start_step: int,
    end_step: int,
    curriculum_type: str = "linear",  # "linear" or "exponential"
    growth_rate: float = 2.0,  # Only used for exponential
):
    """Unified curriculum for all push axes with configurable target velocities.
    
    Args:
        target_velocities: Dict mapping axis names to (min, max) velocity ranges
                          e.g., {"x": (-1.5, 1.5), "y": (-1.5, 1.5), "z": (-0.8, 0.8), 
                                "roll": (-1.5, 1.5), "pitch": (-1.5, 1.5), "yaw": (-1.5, 1.5)}
        curriculum_type: "linear" or "exponential" progression
        growth_rate: Growth rate for exponential curriculum
    
    The return dict is logged to tensorboard.
    """
    if env.common_step_counter < start_step:
        progress = 0.0
        if curriculum_type == "exponential":
            current_scale = 0.05
        else:  # linear
            current_scale = 0.1
    elif env.common_step_counter >= end_step:
        progress = 1.0
        current_scale = 1.0
    else:
        # Calculate progress and scaling
        progress = (env.common_step_counter - start_step) / (end_step - start_step)
        
        if curriculum_type == "exponential":
            start_scale = 0.05
            end_scale = 1.0
            # Exponential growth: scale = start_scale * (end_scale/start_scale)^progress^growth_rate
            current_scale = start_scale * ((end_scale / start_scale) ** (progress ** growth_rate))
        else:  # linear
            start_scale = 0.1
            end_scale = 1.0
            current_scale = start_scale + progress * (end_scale - start_scale)
    
    # Update all axes and collect logging data
    logging_data = {
        "push_curriculum_progress": progress,
        # "push_curriculum_scale": current_scale,
        # "push_curriculum_is_exponential": 1.0 if curriculum_type == "exponential" else 0.0,
    }
    
    
    # Update the push velocity range directly in the event manager
    if hasattr(env.event_manager, 'cfg') and hasattr(env.event_manager.cfg, 'push_robot'):
        if not hasattr(env.event_manager.cfg.push_robot.params, 'velocity_range'):
            env.event_manager.cfg.push_robot.params['velocity_range'] = {}
        
        for axis, target_velocity in target_velocities.items():
            # Calculate current velocity for this axis
            current_velocity = (target_velocity[0] * current_scale, target_velocity[1] * current_scale)
            
            # Update the configuration
            env.event_manager.cfg.push_robot.params['velocity_range'][axis] = current_velocity
            
            # Add per-axis logging
            current_magnitude = max(abs(current_velocity[0]), abs(current_velocity[1]))
            target_magnitude = max(abs(target_velocity[0]), abs(target_velocity[1]))
            
            logging_data.update({
                f"push_{axis}_magnitude": current_magnitude,
                # f"push_{axis}_target_magnitude": target_magnitude,
                # f"push_{axis}_min": current_velocity[0],
                # f"push_{axis}_max": current_velocity[1],
            })
    
    return logging_data



def jump_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    jump_height_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Make the robot jump by setting vertical velocity based on desired jump height.
    
    Uses physics calculation: v = sqrt(2 * g * h) to determine required initial velocity.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to apply jump to
        jump_height_range: (min_height, max_height) in meters for random jump height
        asset_cfg: Asset configuration for the robot
    """
    # Sample random jump height for each environment
    jump_heights = torch.rand(len(env_ids), device=env.device)
    jump_heights = jump_heights * (jump_height_range[1] - jump_height_range[0]) + jump_height_range[0]
    
    gravity_magnitude = abs(env.sim.cfg.gravity[2])
    
    required_velocities = torch.sqrt(2 * gravity_magnitude * jump_heights)
    
    max_velocity = float(torch.max(required_velocities))
    min_velocity = float(torch.min(required_velocities))
    
    velocity_range = {
        "x": (0.0, 0.0),  # No horizontal velocity
        "y": (0.0, 0.0),  # No horizontal velocity  
        "z": (min_velocity, max_velocity),  # Vertical jump velocity
        "roll": (0.0, 0.0),   # No angular velocity
        "pitch": (0.0, 0.0),  # No angular velocity
        "yaw": (0.0, 0.0),    # No angular velocity
    }
    
    # Apply the jump using the existing push_by_setting_velocity function
    mdp.push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)

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


# Latency simulation configuration
# With sim.dt = 0.005 (200Hz), latency steps translate to:
# 0 steps = 0ms (no latency), 1 step = 5ms, 2 steps = 10ms, 3 steps = 15ms, 5 steps = 25ms, 8 steps = 40ms
# 
# Performance guide:
# - (0, 0): No latency - fastest, original speed
# - (0, 1): Binary latency - very fast, ~5% overhead  
# - (1, 3): Low latency - fast, ~10% overhead
# - (1, 5): Medium latency - moderate, ~15% overhead
# - (1, 8): High latency - realistic but slower, ~25% overhead
#
# Real-world sensor latencies:
# - IMU: 1-10ms (gyro/accel), 5-20ms (magnetometer)
# - Vision: 16-33ms (30-60 FPS cameras)
# - Lidar: 50-100ms (mechanical), 10-50ms (solid state)

# Curriculum timing constants
CURRICULUM_START_STEP = 24 * 10   # 10
CURRICULUM_END_STEP = 24 * 100   # 1000

# Command curriculum timing constants
COMMAND_CURRICULUM_START_STEP = 24 * 100   # 1000
COMMAND_CURRICULUM_END_STEP = 24 * 200     # 1500


KBOT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
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

@configclass
class KBotRewards(RewardsCfg):
    """Reward terms for the K-Bot velocity task."""

    # -- base tracking & termination --
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-100.0)
    stay_alive = RewTerm(func=mdp.is_alive, weight=1.0)
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=8.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.85,
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
        weight=0.5,
        params={
            "target_height": 0.15,  # 15cm target height for swing phase
            "std": 0.05,           # Standard deviation for exponential kernel
            "tanh_mult": 2.0,      # Velocity multiplier for swing detection
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
        weight=-0.8,
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

    joint_deviation_ankle = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["dof_left_ankle_02", "dof_right_ankle_02"]),
        },
    )

    joint_deviation_knee = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["dof_left_knee_04", "dof_right_knee_04"]),
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
    
    # foot_collision_explicit = RewTerm(
    #     func=mdp.body_distance_penalty,
    #     weight=-2.0,
    #     params={
    #         "min_distance": 0.3,
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "body_a_names": ["KB_D_501L_L_LEG_FOOT"],  # Left foot
    #         "body_b_names": ["KB_D_501R_R_LEG_FOOT"],  # Right foot
    #     },
    # )
    
    # Action smoothness penalty
    action_acceleration_l2 = RewTerm(
        func=action_acceleration_l2,
        weight=-0.1,
    )

    # Foot contact force penalty - L2 penalty above threshold
    foot_contact_force_l2 = RewTerm(
        func=contact_forces_l2_penalty,
        weight=-1.0e-9,
        params={
            "threshold": 380.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"],
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
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"])},
        )
        
        # Body poses for important body parts (privileged state info)
        body_poses = ObsTerm(
            func=mdp.body_pose_w,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["base", "KB_D_501L_L_LEG_FOOT", "KB_D_501R_R_LEG_FOOT"])},
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
        base_pos = ObsTerm(func=mdp.base_pos_z, noise=Unoise(n_min=-0.0001, n_max=0.0001))
        
        # Root state information (privileged)
        root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001))
        root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, noise=Unoise(n_min=-0.0001, n_max=0.0001))

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        # observation terms (order preserved)
        projected_gravity = ObsTerm(
            func=imu_projected_gravity_with_latency,
            params={
                "asset_cfg": SceneEntityCfg("imu"),
                "latency_steps_range": (0, 1),
                "buffer_size": 4,
            },
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, noise=Unoise(n_min=-math.radians(5), n_max=math.radians(5))
        )
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )


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
class KBotCurriculumCfg(CurriculumCfg):
    """Curriculum terms for the K-Bot locomotion task."""

    # Unified Push Robot Curriculum - Linear progression for linear velocities
    push_linear_velocities = CurrTerm(
        func=unified_push_curriculum,
        params={
            "target_velocities": {
                "x": (-1.5, 1.5),
                "y": (-1.5, 1.5),
                "z": (-0.8, 0.8),
            },
            "start_step": CURRICULUM_START_STEP,
            "end_step": CURRICULUM_END_STEP,
            "curriculum_type": "linear",
        }
    )
    
    # Unified Push Robot Curriculum - Exponential progression for angular velocities
    push_angular_velocities = CurrTerm(
        func=unified_push_curriculum,
        params={
            "target_velocities": {
                "roll": (-1.5, 1.5),
                "pitch": (-1.5, 1.5),
                "yaw": (-1.5, 1.5),
            },
            "start_step": CURRICULUM_START_STEP,
            "end_step": CURRICULUM_END_STEP,
            "curriculum_type": "exponential",
            "growth_rate": 1.5,
        }
    )

    # Unified Command Range Curriculum
    command_ranges = CurrTerm(
        func=unified_command_curriculum,
        params={
            "initial_ranges": {
                "lin_vel_x": (-0.5, 0.5),
                "lin_vel_y": (-0.5, 0.5),
                "ang_vel_z": (-0.1, 0.1),
            },
            "target_ranges": {
                "lin_vel_x": (-2.0, 2.0),
                "lin_vel_y": (-1.5, 1.5),
                "ang_vel_z": (-1.0, 1.0),
            },
            "start_step": COMMAND_CURRICULUM_START_STEP,
            "end_step": COMMAND_CURRICULUM_END_STEP,
        }
    )


@configclass
class KBotRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: KBotRewards = KBotRewards()
    observations: KBotObservations = KBotObservations()
    curriculum: KBotCurriculumCfg = KBotCurriculumCfg()
    # terminations: KBotTerminations = KBotTerminations()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.actions.action_history_length = 3

        self.scene.terrain.terrain_generator = KBOT_ROUGH_TERRAINS_CFG

        randomize_every_reset = True
        # randomize_every_reset = False

        # Scene
        self.scene.robot = KBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"

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

        # # Randomization
        self.events.reset_robot_joints.func=mdp.reset_joints_by_offset
        self.events.reset_robot_joints.params["position_range"] = (-math.radians(20), math.radians(20))
        self.events.reset_robot_joints.params["velocity_range"] = (-0.2, 0.2)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["base"]
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
                "pitch": (-math.radians(20), math.radians(20)),
                "roll": (-math.radians(20), math.radians(20)),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        self.events.jump_robot = EventTerm(
            func=jump_robot,
            mode="interval",
            interval_range_s=(15.0, 30.0),
            params={
                "jump_height_range": (0.1, 0.3),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Start with gentle pushes - curriculum will progressively increase these
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.08, 0.08),    # Start at 10% of target intensity
            "y": (-0.08, 0.08),    # Start at 10% of target intensity
            "z": (-0.015, 0.015),  # Start at 5% of target intensity  
            "roll": (-0.025, 0.025),   # Start at 5% of target intensity
            "pitch": (-0.025, 0.025),  # Start at 5% of target intensity
            "yaw": (-0.025, 0.025),    # Start at 5% of target intensity
        }

        self.events.push_robot.interval_range_s = (0.5, 15.0)

        self.events.add_base_mass = None

        self.events.randomize_link_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot",
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
                    ],),
                "mass_distribution_params": (0.7, 1.3),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )

        self.events.physics_material.params["static_friction_range"] = (0.3, 0.9)
        self.events.physics_material.params["dynamic_friction_range"] = (0.2, 0.7)
        self.events.physics_material.params["restitution_range"] = (0.0, 0.1)
        self.events.physics_material.params["make_consistent"] = True  # Ensure dynamic friction is always less than static friction

        self.events.randomize_joint_parameters = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "armature_distribution_params": (0.5, 4.0),
                "friction_distribution_params": (0.5, 4.0),
                "operation": "scale",
            },
        )

        self.events.randomize_imu_mount = EventTerm(
            func=randomize_imu_mount,
            mode="reset",
            params={
                "sensor_cfg": SceneEntityCfg("imu"),
                "pos_range": {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (-0.03, 0.03),
                },
                "rot_range": {
                    "roll": (-math.radians(2.0), math.radians(2.0)),
                    "pitch": (-math.radians(2.0), math.radians(2.0)),
                    "yaw": (-math.radians(2.0), math.radians(2.0)),
                },
            },
        )

        if randomize_every_reset:
            self.events.physics_material.mode = "reset"
            self.events.randomize_joint_parameters.mode = "reset"

        # Rewards
        self.rewards.lin_vel_z_l2.weight = -0.01
        self.rewards.undesired_contacts = None
        self.rewards.flat_orientation_l2.params["asset_cfg"] = SceneEntityCfg("imu")
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.action_rate_l2.weight = -0.1
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
        self.rewards.dof_torques_l2.weight = -2.5e-6 # -1.5e-7
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
        
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)  # Curriculum will gradually increase to (-2.0, 2.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)  # Curriculum will gradually increase to (-1.5, 1.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.1, 0.1)    # Curriculum will gradually increase to (-1.0, 1.0)

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

        # self.commands.base_velocity.ranges.lin_vel_x = (1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.curriculum = None
        # self.events = None
