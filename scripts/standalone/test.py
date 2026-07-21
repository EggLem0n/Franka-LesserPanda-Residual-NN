# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

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

from LessP.additional.record_simulation import SimulationRecorder, get_camera_cfg, get_object_cfg, ENABLED, VIDEO_DIR

from isaaclab_assets import FRANKA_PANDA_CFG


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Design the scene with external torque PD control."""
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.7405)),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )
    table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0, 0.0], rot=[1.0, 0, 0, 0.0]),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd",
                                   scale=(1.5, 2.0, 1.0)),
    )
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
    # 내장 PD 비활성화 → 외부 토크 제어
    for _name in robot.actuators:
        robot.actuators[_name].stiffness = 0.0
        robot.actuators[_name].damping   = 0.0
    del _name

    record_camera = get_camera_cfg()
    object        = get_object_cfg()


# ── 목표 조인트 포지션 (단위: 라디안) ────────────────────────────────────────
# 원하는 값으로 수정하세요. 지정하지 않은 조인트는 초기 자세를 유지합니다.
TARGET_JOINT_POS = {

    "panda_joint1": 0.0,
    "panda_joint2": -0.569,
    "panda_joint3": 0.0,
    "panda_joint4": -2.810,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint.*": 0.04,
    # "panda_joint1": np.radians(0.0),
    # "panda_joint2": np.radians(0.0),
    # "panda_joint3": np.radians(0.0),
    # "panda_joint4": np.radians(0.0),
    # "panda_joint5": np.radians(0.0),
    # "panda_joint6": np.radians(0.0),
    # "panda_joint7": np.radians(0.0),
    # "panda_finger_joint1": np.radians(0.0),
    # "panda_finger_joint2": np.radians(0.0),
}

# ── PD 게인 (조인트 순서: joint1~7, finger1, finger2) ────────────────────────
# τ = Kp*(q_target - q) - Kd*q_dot
KP = [80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0,  0.0,  0.0]
KD = [ 40.0,  40.0,  40.0,  40.0,  40.0,  40.0,  40.0,  0.0,  0.0]


class PDController:
    def __init__(self, kp: list, kd: list, num_envs: int, device: str):
        self.kp = torch.tensor(kp, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1)
        self.kd = torch.tensor(kd, dtype=torch.float32, device=device).unsqueeze(0).expand(num_envs, -1)

    def compute(self, q: torch.Tensor, q_dot: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
        return self.kp * (q_target - q) - self.kd * q_dot


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, recorder: SimulationRecorder | None = None):
    robot   = scene["robot"]
    sim_dt  = sim.get_physics_dt()
    device  = scene.device
    n_envs  = scene.num_envs

    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    # 초기 상태
    robot.update(sim_dt)
    default_joint_pos  = robot.data.default_joint_pos.clone()
    default_joint_vel  = robot.data.default_joint_vel.clone()
    default_root_state = robot.data.default_root_state.clone()

    target_pos = default_joint_pos.clone()
    for i, name in enumerate(robot.data.joint_names):
        if name in TARGET_JOINT_POS:
            target_pos[:, i] = TARGET_JOINT_POS[name]

    target_for_plot = target_pos[0, robot_entity_cfg.joint_ids].cpu().numpy()

    robot.write_root_state_to_sim(default_root_state)
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.write_data_to_sim()
    robot.reset()
    robot.update(sim_dt)

    controller = PDController(KP, KD, n_envs, device)

    sim_len = 10.0
    t = 0.0
    log_joint_pos = []

    if recorder:
        print("[INFO]: 영상 녹화 활성화 (60 FPS)")

    while simulation_app.is_running() and t <= sim_len:
        q     = robot.data.joint_pos
        q_dot = robot.data.joint_vel
        torque = controller.compute(q, q_dot, target_pos)

        robot.set_joint_effort_target(torque)
        robot.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_dt)
        robot.update(sim_dt)

        log_joint_pos.append(robot.data.joint_pos[0, robot_entity_cfg.joint_ids].cpu().numpy())

        if recorder:
            recorder.capture_frame(scene, sim_dt)

        t += sim_dt

    if recorder:
        recorder.save()

    # ── 플롯 ──────────────────────────────────────────────────────────────────
    joint_names        = [robot.data.joint_names[i] for i in robot_entity_cfg.joint_ids]
    log_joint_pos_deg  = np.degrees(np.array(log_joint_pos))
    target_for_plot_deg = np.degrees(target_for_plot)
    t_arr              = np.linspace(0, t, len(log_joint_pos_deg))
    n_joints           = log_joint_pos_deg.shape[1]

    fig, axes = plt.subplots(n_joints, 1, figsize=(10, 2 * n_joints), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t_arr, log_joint_pos_deg[:, i], label="actual")
        ax.axhline(y=target_for_plot_deg[i], color='red', linestyle='--', linewidth=1.2, label="target")
        ax.text(t_arr[-1], target_for_plot_deg[i], f" {target_for_plot_deg[i]:.1f}°",
                color='red', va='bottom', ha='right', fontsize=7)
        ax.set_ylabel(f"{joint_names[i]} [deg]")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True)
    axes[-1].set_xlabel("t [s]")
    plt.tight_layout()
    plt.show()


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0/60.0, device=args_cli.device)
    sim     = SimulationContext(sim_cfg)
    sim.set_camera_view([2.5, 2.5, 4.0], [0.0, 0.0, 0.0])

    scene_cfg = RobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene     = InteractiveScene(scene_cfg)
    sim.reset()

    recorder = SimulationRecorder(VIDEO_DIR) if ENABLED else None
    if recorder:
        recorder.sync_camera_to_viewport()

    print("[INFO]: Setup complete...")
    run_simulator(sim, scene, recorder)


if __name__ == "__main__":
    main()
    simulation_app.close()
