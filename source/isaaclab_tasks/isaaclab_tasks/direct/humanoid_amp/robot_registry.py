from dataclasses import dataclass
from isaaclab.assets import ArticulationCfg
import os
# path to the motions directory (avoid circular import)
MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")
from isaaclab_assets.robots.humanoid_28 import HUMANOID_28_CFG
from isaaclab_assets.robots.kbot import KBOT_CFG

@dataclass(frozen=True)
class RobotConfig:
    """Robot-specific configuration including observation and action space sizes."""
    observation_space: int
    action_space: int
    state_space: int = 0
    amp_observation_space: int = None  # will default to observation_space if None
    
    def __post_init__(self):
        if self.amp_observation_space is None:
            object.__setattr__(self, 'amp_observation_space', self.observation_space)

@dataclass(frozen=True)
class RobotAMPDescriptor:
    name: str
    cfg: ArticulationCfg # asset cfg 
    motion_file: str              
    reference_body: str           
    key_bodies: list[str] # 4 limbs
    robot_config: RobotConfig

ROBOTS = {}
def register(desc: RobotAMPDescriptor):
    ROBOTS[desc.name] = desc



# register things I will definitely use

# Humanoid configs (28 DOFs)
# obs = 2*dof + 1 (height) + 6 (tangent+normal) + 3 (lin_vel) + 3 (ang_vel) + 3*4 (key_bodies)
# obs = 2*28 + 1 + 6 + 3 + 3 + 12 = 56 + 25 = 81
HUMANOID_CONFIG = RobotConfig(
    observation_space=81,
    action_space=28,
    state_space=0,
    amp_observation_space=81,
)

register(
    RobotAMPDescriptor(
        name="default_humanoid_walk",
        cfg=HUMANOID_28_CFG,
        motion_file=os.path.join(MOTIONS_DIR, "humanoid_walk.npz"),
        reference_body="torso",
        key_bodies=["right_hand", "left_hand", "right_foot", "left_foot"],
        robot_config=HUMANOID_CONFIG,
    )
)

register(
    RobotAMPDescriptor(
        name="default_humanoid_run",
        cfg=HUMANOID_28_CFG,
        motion_file=os.path.join(MOTIONS_DIR, "humanoid_run.npz"),
        reference_body="torso",
        key_bodies=["right_hand", "left_hand", "right_foot", "left_foot"],
        robot_config=HUMANOID_CONFIG,
    )
)

register(
    RobotAMPDescriptor(
        name="default_humanoid_dance",
        cfg=HUMANOID_28_CFG,
        motion_file=os.path.join(MOTIONS_DIR, "humanoid_dance.npz"),
        reference_body="torso",
        key_bodies=["right_hand", "left_hand", "right_foot", "left_foot"],
        robot_config=HUMANOID_CONFIG,
    )
)

# KBot configs (20 DOFs). NOTE: MIGHT HAVE TO DEBUG THIS
# obs = 2*dof + 1 (height) + 6 (tangent+normal) + 3 (lin_vel) + 3 (ang_vel) + 3*4 (key_bodies)
# obs = 2*20 + 1 + 6 + 3 + 3 + 12 = 40 + 25 = 65
KBOT_CONFIG = RobotConfig(
    observation_space=65,
    action_space=20,
    state_space=0,
    amp_observation_space=65,
)

register(
    RobotAMPDescriptor(
        name="kbot_walk",
        cfg=KBOT_CFG,
        motion_file=os.path.join(MOTIONS_DIR, "kbot_humanoid_walk.npz"),
        reference_body="base", 
        key_bodies=["KB_C_501X_Right_Bayonet_Adapter_Hard_Stop", "KB_C_501X_Left_Bayonet_Adapter_Hard_Stop", "KB_D_501R_R_LEG_FOOT", "KB_D_501L_L_LEG_FOOT"],
        robot_config=KBOT_CONFIG,
    )
)


__all__ = [
    "RobotAMPDescriptor",
    "RobotConfig",
    "ROBOTS",
]