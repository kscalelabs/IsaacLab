# Tips
To run with root: `export OMNI_KIT_ALLOW_ROOT=1`


# Training
```
conda activate isaaclab
sudo ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Rough-H1-v0 --num_envs 4096 --headless --video --enable_cameras

sudo ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Rough-Stompy-v0 --num_envs 4096 --headless --video --enable_cameras
```

# Eval
```
sudo ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Rough-H1-v0
sudo ./isaac_lab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Rough-Stompy-v0
```

# Random
Multinode training:
https://isaac-sim.github.io/IsaacLab/source/features/multi_gpu.html
