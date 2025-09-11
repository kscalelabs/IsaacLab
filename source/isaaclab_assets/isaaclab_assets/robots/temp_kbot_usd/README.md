
# Instructions 

For importing the kbot into Isaaclab


## URDF Importer

Instead of using the GUI option, use the script with command 

```bash
# cd into Isaaclab folder
./isaaclab.sh -p scripts/tools/convert_urdf.py /path/to/robot.urdf /path/to/desired/output/robot.usd
```

The reason for this is that you need to NOT merge the joints, because otherwise the imu will get merged into the base and not have a "RigidBodyAPI" which is needed.

## Update training script 

Note that you will need to change the names of anything referencing the geoms in the train config ,for example in:

`IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/kbot/rough_rnn_env_cfg.py`

```python
# Physics material randomization (friction with the floor)
self.events.physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg(
            "robot", body_names=["LFootBushing_GPF_1517_12", "RFootBushing_GPF_1517_12"]
        ),
        "static_friction_range": (0.1, 2.0),
        "dynamic_friction_range": (0.1, 2.0),
        "restitution_range": (0.0, 0.1),
        "num_buckets": 64,
        "make_consistent": True, # Ensure dynamic friction is always less than static friction
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
                "KD_B_102B_TORSO_BTM",
                "KD_D_102L_L_Hip_Yoke_Drive",
                "KD_C_101L_ShldYokeDrive",
                "KD_D_102R",
                "KC_C_101R_ShldYokeDrive",
                "L_Hip_Roll_RS03",
                "L_Hip_Roll_RS03_2",
                "RS03_4",
                "RS03_3",
                "KD_D_301L_L_Femur_Lower_Drive",
                "KD_C_301L_LowerBicepDrive",
                "KD_D_301R",
                "KC_C_202R",
                "KD_D_401L_L_Shin_Drive",
                "KC_C_401L_Up_Forearm_Drive",
                "KD_D_401R",
                "KC_C_401R_R_UpForearmDrive",
                "LFootBushing_GPF_1517_12",
                "PRT0001_2",
                "RFootBushing_GPF_1517_12",
                "PRT0001",
            ],
        ),
        "mass_distribution_params": (0.8, 1.2),
        "operation": "scale",
        "distribution": "uniform",
        "recompute_inertia": True,
    },
)
```


