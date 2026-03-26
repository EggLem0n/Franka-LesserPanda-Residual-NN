# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_articulation.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import matplotlib.pyplot as plt
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG
from isaaclab_assets import CARTPOLE_CFG  # isort:skip

@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene Implicit Actuators on the robot."""
    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.7405)),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    # Table
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0.0], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
                                   scale=(1.5, 2.0, 1.0)),
    )
    # robot
    robot = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        )
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.0, 1.0, 1.0),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.5, 0.2, 0.0), rot=(1.0, 0.0, 0.0, 0.0)
        ),
    )


def run_simulator(sim: sim_utils.SimulationContext, scene : InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    robot = scene["robot"]

    # 초기 상태 명시적으로 시뮬레이터에 반영
    robot.update(sim_dt)
    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    #robot.write_data_to_sim()
    robot.reset()

    sim_len = 10.0
    t = 0.0
    log_joint_pos = []

    # Simulation loop
    while simulation_app.is_running() and t <= sim_len:
        sim.step()
        scene.update(sim_dt)
        robot.update(sim_dt)
        log_joint_pos.append(robot.data.joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy())
        t += sim_dt

    # Plot
    # joint 이름 가져오기
    joint_names = [robot.data.joint_names[i] for i in
    robot_entity_cfg.joint_ids]
    log_joint_pos = np.array(log_joint_pos)
    t_arr = np.linspace(0, t, len(log_joint_pos))
    n_joints = log_joint_pos.shape[1]
    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2 * n_joints), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_arr, log_joint_pos[:, i])
        ax.set_ylabel(joint_names[i])
        ax.grid(True)
    axes[-1].set_xlabel("t [s]")
    plt.tight_layout()
    plt.show()

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 4.0], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
