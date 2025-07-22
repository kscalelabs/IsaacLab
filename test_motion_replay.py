#!/usr/bin/env python3
"""
Simple script to replay motion dataset and verify joint mappings.
"""

import time
import numpy as np
import torch
from isaaclab.app import AppLauncher

# Launch Isaac Sim
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext, SimulationCfg, DomeLightCfg, GroundPlaneCfg
from isaaclab.sim.spawners import spawn_ground_plane
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab_tasks.direct.humanoid_amp.motions import MotionLoader
from isaaclab_tasks.direct.humanoid_amp.robot_registry import ROBOTS

def main():
    # Choose which robot to test
    robot_name = "kbot_walk"  # or "default_humanoid_walk"
    
    # Get robot descriptor
    desc = ROBOTS[robot_name]
    print(f"Testing robot: {robot_name}")
    print(f"Motion file: {desc.motion_file}")
    print(f"Reference body: {desc.reference_body}")
    print(f"Key bodies: {desc.key_bodies}")
    
    # Setup simulation
    sim_cfg = SimulationCfg(device="cuda")
    sim = SimulationContext(sim_cfg)
    
    # Add ground plane
    spawn_ground_plane("/World/ground", cfg=GroundPlaneCfg())
    # Add a dome light
    dome_light_cfg = DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    dome_light_cfg.func("/World/Light", dome_light_cfg)
    
    # Set camera view
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 1.0])
    
    # Setup scene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    
    # Spawn robot
    robot_cfg = desc.cfg.replace(prim_path="/World/envs/env_0/Robot")
    robot = Articulation(robot_cfg)
    scene.articulations["robot"] = robot
    
    # Play the scene to initialize all assets, then reset
    # This is the correct initialization order.
    sim.play()
    scene.reset()
    
    # Load motion data
    motion_loader = MotionLoader(motion_file=desc.motion_file, device="cuda")
    
    print(f"\nMotion info:")
    print(f"- Duration: {motion_loader.duration:.2f} sec")
    print(f"- Frames: {motion_loader.num_frames}")
    print(f"- DOF names in motion: {motion_loader.dof_names}")
    print(f"- Body names in motion: {motion_loader.body_names}")
    
    print(f"\nRobot info:")
    print(f"- DOF names in robot: {robot.data.joint_names}")
    print(f"- Body names in robot: {robot.data.body_names}")
    
    # Check mapping
    try:
        motion_dof_indexes = motion_loader.get_dof_index(robot.data.joint_names)
        print(f"\nMapping successful! Motion DOF indexes: {motion_dof_indexes}")
    except Exception as e:
        print(f"\nMapping FAILED: {e}")
        print("This means joint names don't match between motion file and robot USD.")
        return
    
    # Check body mapping
    try:
        ref_body_idx = motion_loader.get_body_index([desc.reference_body])[0]
        key_body_idx = motion_loader.get_body_index(desc.key_bodies)
        print(f"Reference body '{desc.reference_body}' found at index: {ref_body_idx}")
        print(f"Key bodies found at indexes: {key_body_idx}")
    except Exception as e:
        print(f"Body mapping FAILED: {e}")
        return
    
    # Play simulation
    scene.reset()
    
    # Sample some motion frames to replay
    num_samples = len(motion_loader.dof_positions)
    times = np.linspace(0, motion_loader.duration, num_samples)
    
    print(f"\nReplaying {num_samples} frames from motion...")
    
    # Track the current frame
    frame_idx = 0
    
    # Keep running and rendering
    while simulation_app.is_running():
        # Step the simulation
        # sim.step()
        # update the scene
        # scene.update(dt=sim.get_physics_dt())
    
        # on the first frame, reset the robot to the initial motion pose
        # if sim.current_time == 0:
        #     scene.reset()
    
        # during the replay, apply the motion data
        if frame_idx < num_samples:
            # sample motion at the current time
            t = times[frame_idx]
            dof_pos, dof_vel, body_pos, body_rot, _, _ = motion_loader.sample(
                num_samples=1, times=np.array([t])
            )
    
            # map to robot joints
            robot_dof_pos = dof_pos[0, motion_dof_indexes]
            robot_dof_vel = dof_vel[0, motion_dof_indexes]
    
            # get reference body pose from motion
            ref_body_idx = motion_loader.get_body_index([desc.reference_body])[0]
            ref_body_pos = body_pos[0, ref_body_idx]
            ref_body_rot = body_rot[0, ref_body_idx]
    
            # apply to robot
            robot.write_joint_state_to_sim(
                position=robot_dof_pos.unsqueeze(0),
                velocity=robot_dof_vel.unsqueeze(0),
            )
            robot.write_root_link_pose_to_sim(torch.cat([ref_body_pos, ref_body_rot]).unsqueeze(0))
    
            # print progress
            if frame_idx % 10 == 0:
                print(f"Frame {frame_idx}/{num_samples}, time: {t:.2f}s")
            
            frame_idx += 1
        # after replay, print a message once
        elif frame_idx == num_samples:
            print("Motion replay complete! The robot is holding its final pose.")
            frame_idx += 1
    
        # update the app to render the viewport at 60 Hz
        simulation_app.update()
        time.sleep(1.0 / 60.0)

if __name__ == "__main__":
    main() 