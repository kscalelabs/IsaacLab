# Tips
To run with root: `export OMNI_KIT_ALLOW_ROOT=1`

# Creating .usd
```
1. Open Isaac Sim in NVIDIA Omniverse
2. Create a new scene
3. Install URDF Importer, or enable it if it's already installed
4. Import the URDF file, making sure that an output directory is set
5. .usd file will be saved in the same directory as the URDF
```


# Training
```
conda activate isaaclab
sudo ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Flat-Dora-v0 --num_envs 4096 --headless --video --enable_cameras

sudo ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task Isaac-Velocity-Rough-Dora-v0 --num_envs 4096 --headless --video --enable_cameras
```

# Eval
```
sudo ./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py --task Isaac-Velocity-Rough-Dora-v0 --num_envs=16
```

# Random
Multinode training:
https://isaac-sim.github.io/IsaacLab/source/features/multi_gpu.html
