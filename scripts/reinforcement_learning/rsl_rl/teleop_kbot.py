# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a keyboard teleoperation with Isaac Lab manipulation environments."""

"""Launch Isaac Sim Simulator first."""

import argparse
from collections.abc import Callable

from isaaclab.app import AppLauncher
import numpy as np


# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
import cli_args  # isort: skip
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import torch

import omni.log

from isaaclab.utils.assets import retrieve_file_path

from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import time

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg

import omni.replicator.core as rep
import cv2
import subprocess
import socket
import json
import threading
from pathlib import Path

def open_ffmpeg_stream_process():
    args = (
        "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
        "rgb24 -s 1280x720 -i pipe:0 -pix_fmt yuv420p "
        "-f mpegts udp://127.0.0.1:8554"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)

# Alternative with higher quality settings
def open_ffmpeg_stream_process_high_quality():
    args = (
        "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
        "rgb24 -s 1280x720 -i pipe:0 -c:v libx264 -preset slow "
        "-crf 20 -pix_fmt yuv420p -b:v 3M -maxrate 4M -bufsize 8M "
        "-profile:v high -level 4.1 -g 60 -keyint_min 30 "
        "-f mpegts udp://127.0.0.1:8554"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)

command_data = {
    'joints': {
        '21': 0.0,
        '22': 0.0,
        '23': 0.0,
        '24': 0.0,
        '25': 0.0,
        '11': 0.0,
        '12': 0.0,
        '13': 0.0,
        '14': 0.0,
        '15': 0.0
    }
}
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind(("0.0.0.0", 8888))

def update_data_thread():
    global command_data
    while True:
        data, addr = sock.recvfrom(512)
        message = data.decode("utf-8")
        command_data.update(json.loads(message))

def main() -> None:
    global command_data
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    threading.Thread(target=update_data_thread, daemon=True).start()

    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)

    task_name = args_cli.task.split(":")[-1]
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)


    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    env.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment

    ffmpeg_process = open_ffmpeg_stream_process()

    left_cam_path = "/World/envs/env_0/Robot/KD_B_102B_TORSO_BTM/Camera"
    right_cam_path = "/World/envs/env_0/Robot/KD_B_102B_TORSO_BTM/Camera_01"
    resolution = (1280,1080)#env.env.cfg.viewer.resolution

    # Create render products
    render_product_left = rep.create.render_product(left_cam_path, resolution)
    render_product_right = rep.create.render_product(right_cam_path, resolution)

    # Create and attach RGB annotator
    rgb_annotator_left = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator_right = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb_annotator_left.attach([render_product_left])
    rgb_annotator_right.attach([render_product_right])
    videos_dir = Path("/home/miller/IsaacLab/videos")
    videos_dir.mkdir(parents=True, exist_ok=True)
    left_dir = videos_dir / "left"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir = videos_dir / "right"
    right_dir.mkdir(parents=True, exist_ok=True)
    i = 0
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping

            # set reasonable wrist targets
            # observations go left, right (positive y, negative y)
            # obs[:, -14:-7] = torch.Tensor([ # xyz, quat
            #     [0.2, 0.1, 0.1, 1.0, 0, 0, 0]
            # ])
            # obs[:, -7:] =  torch.Tensor([ # xyz, quat
            #     [0.2, -0.1, 0.1, 1.0, 0, 0, 0]
            # ])
            # obs[:,9:12] = torch.zeros(3) # velocity command
            # command[0:3] = command_data.get('velocity', [0.0, 0.0, 0.0])
            # command[3:10] = command_data.get('right_ee', [0.0]*7)
            # command[10:17] = command_data.get('left_ee', [0.0]*7)
            # obs[:, 40:40+17] = command
            actions = policy(obs)
            # command_data has absolute angles, but gym environment wants them relative to starting positions
            # "dof_right_shoulder_pitch_03": 0.0,
            # "dof_right_shoulder_roll_03": math.radians(-10.0),
            # "dof_right_shoulder_yaw_02": 0.0,
            # "dof_right_elbow_02": math.radians(90.0),
            # "dof_right_wrist_00": 0.0,
            actions[:, (3,7,11,15,19)] = torch.deg2rad(2*torch.Tensor([command_data['joints'][k] for k in ['21', '22', '23', '24', '25']]).to(device=actions.device))
            actions[:, (7, 15)] -= 2*torch.deg2rad(torch.Tensor([-10, 90]).to(device=actions.device))

            actions[:, (1, 5, 9, 13, 17)] = torch.deg2rad(2*torch.Tensor([command_data['joints'][k] for k in ['11', '12', '13', '14', '15']]).to(device=actions.device))
            actions[:,13] = -actions[:,13]
            actions[:,1] = -actions[:,1]
            actions[:, (5, 13)] -= 2*torch.deg2rad(torch.Tensor([10, -90]).to(device=actions.device))

            # actions[:, 7] -= np.deg2rad(-10)
            # actions[: 15] -= np.deg2rad(90)
            rgb_data_left = rgb_annotator_left.get_data()
            rgb_data_right = rgb_annotator_right.get_data()
            # rgb_data shape: (2, H, W, 4) or (2, H, W, 3) depending on annotator
            # convert to numpy array
            rgb_data_left = np.frombuffer(rgb_data_left, dtype=np.uint8).reshape(*rgb_data_left.shape)
            rgb_data_right = np.frombuffer(rgb_data_right, dtype=np.uint8).reshape(*rgb_data_right.shape)
            # return the rgb data
            # note: initially the renerer is warming up and returns empty data
            if rgb_data_left.size != 0 and rgb_data_right.size != 0:
                frame_left = rgb_data_left[:, :, :3]
                frame_right = rgb_data_right[:, :, :3]
                cv2.imwrite(left_dir / f"{i}.png", cv2.cvtColor(frame_left, cv2.COLOR_RGB2BGR))
                cv2.imwrite(right_dir / f"{i}.png", cv2.cvtColor(frame_right, cv2.COLOR_RGB2BGR))
                # ffmpeg_process.stdin.write(frame_left.tobytes())
            # env stepping
            i+=1
            obs, _, _, _ = env.step(actions)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    omni.log.info("Environment closed")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

'''
#Example video client
import threading
import queue
import cv2

q = queue.Queue()

def receive():
    cap = cv2.VideoCapture('udp://@127.0.0.1:8554?buffer_size=65535&pkt_size=65535&fifo_size=65535')
    ret, frame = cap.read()
    q.put(frame)
    while ret:
        ret, frame = cap.read()
        q.put(frame)

def display():
    while True:
        if q.empty() != True:
            frame = q.get()
            cv2.imshow('Video', frame)

        k = cv2.waitKey(1) & 0xff
        if k == 27:  # press 'ESC' to quit
            break

tr = threading.Thread(target=receive, daemon=True)
td = threading.Thread(target=display)

tr.start()
td.start()

td.join()
'''