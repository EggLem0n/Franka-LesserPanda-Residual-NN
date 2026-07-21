from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG
from isaaclab.actuators import IdealPDActuatorCfg

FRANKA_PANDA_ZERO_SHOT_CFG = FRANKA_PANDA_CFG.replace(
    actuators={
        "panda_shoulder": IdealPDActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=87.0,          # explicit actuator uses this parameter not effort_limit_sim
            velocity_limit=2.1750,      # explicit actuator uses this parameter not effort_velocity_sim
            stiffness=200.0,             # This value is based on the original 'franka.py' gain settings; however, it will be overridden to zero when switching to a PINN actuator drive.
            damping=28.0,                # This value is based on the original 'franka.py' gain settings; however, it will be overridden to zero when switching to a PINN actuator drive.
            armature=0.1,               # This value is based on DeepMind menagerie 'panda.xml'
            friction=0,                 # PINN will handle this!
            dynamic_friction=0,         # PINN will handle this!
            viscous_friction=0,         # PINN will handle this!
        ),
        "panda_forearm": IdealPDActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=12.0,          # explicit actuator uses this parameter not effort_limit_sim
            velocity_limit=2.6100,      # explicit actuator uses this parameter not effort_velocity_sim
            stiffness=150.0,             # This value is based on the original 'franka.py' gain settings; however, it will be overridden to zero when switching to a PINN actuator drive.
            damping=24.0,                # This value is based on the original 'franka.py' gain settings; however, it will be overridden to zero when switching to a PINN actuator drive.
            armature=0.1,               # This value is based on DeepMind menagerie 'panda.xml'
            friction=0,                 # PINN will handle this!
            dynamic_friction=0,         # PINN will handle this!
            viscous_friction=0,         # PINN will handle this!
        ),
    },
)


