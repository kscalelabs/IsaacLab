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
import cv2
import subprocess

def open_ffmpeg_stream_process():
    args = (
        "ffmpeg -re -stream_loop -1 -f rawvideo -pix_fmt "
        "rgb24 -s 1280x720 -i pipe:0 -pix_fmt yuv420p "
        "-f mpegts udp://127.0.0.1:8554"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def main() -> None:
    """
    Run keyboard teleoperation with Isaac Lab manipulation environment.

    Creates the environment, sets up teleoperation interfaces and callbacks,
    and runs the main simulation loop until the application is closed.

    Returns:
        None
    """
    # sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # sock.bind(("0.0.0.0", 1234))

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


    # Monkeypatch env to have camera
    # env.env.cfg.scene.tiled_camera = TiledCameraCfg(
    #     prim_path="/World/envs/env_0/Robot/KD_B_102B_TORSO_BTM/Camera",
    #     data_types=["rgb"],
    # )
    # env.env.cfg.observations.policy.image = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("tiled_camera"), "data_type": "rgb"})
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
    env.env.cfg.viewer.cam_prim_path = "/World/envs/env_0/Robot/KD_B_102B_TORSO_BTM/Camera"
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping

            # data, addr = sock.recvfrom(512)
            # message = data.decode("utf-8")
            # command_data: dict = json.loads(message)

            command = torch.zeros(17)
            # set reasonable wrist targets
            command[3:10] = torch.Tensor([ # xyz, quat
                [0.2, -0.1, 0, 1.0, 0, 0, 0]
            ])
            command[10:17] =  torch.Tensor([ # xyz, quat
                [0.2, 0.1, 0, 1.0, 0, 0, 0]
            ])
            # command[0:3] = command_data.get('velocity', [0.0, 0.0, 0.0])
            # command[3:10] = command_data.get('right_ee', [0.0]*7)
            # command[10:17] = command_data.get('left_ee', [0.0]*7)
            obs[:, -17:] = command
            actions = policy(obs)
            frame = env.env.render()
            ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())
            # env stepping
            obs, _, _, _ = env.step(actions)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

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