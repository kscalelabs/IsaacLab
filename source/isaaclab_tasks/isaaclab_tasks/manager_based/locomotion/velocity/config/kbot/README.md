# Commands

To import urdf to USD, see the temp_kbot_usd folder and the README there.
```bash
IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/temp_kbot_usd
```

To run training:
```bash
# cd IsaacLab
./isaaclab.sh     -p scripts/reinforcement_learning/rsl_rl/train.py     --task Isaac-Velocity-Rough-Kbot-RNN-v0 --headless
```

To export to kinfer:
```bash
# cd IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/export.py   --task=Isaac-Velocity-Rough-Kbot-RNN-v0      --headless --checkpoint /path/to/IsaacLab/logs/rsl_rl/kbot_rough_rnn/2025-09-10_00-22-32/model_9450.pt