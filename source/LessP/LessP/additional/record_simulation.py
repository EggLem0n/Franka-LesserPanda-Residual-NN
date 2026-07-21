"""시뮬레이션 영상 녹화 모듈.

AppLauncher 초기화 전후 모두 import 가능합니다.
- ENABLED, VIDEO_DIR 등의 설정 상수는 AppLauncher 전에 사용 가능
- SimulationRecorder, get_camera_cfg()는 AppLauncher 이후에만 호출 가능

사용 예시 (test.py):
    # AppLauncher 전
    from LessP.additional.record_simulation import ENABLED, VIDEO_DIR
    args_cli.enable_cameras = ENABLED

    # AppLauncher 후
    from LessP.additional.record_simulation import SimulationRecorder, get_camera_cfg

    @configclass
    class MySceneCfg(InteractiveSceneCfg):
        record_camera = get_camera_cfg()

    recorder = SimulationRecorder(VIDEO_DIR) if ENABLED else None
    if recorder:
        recorder.sync_camera_to_viewport()

    if recorder:
        recorder.capture_frame(scene, sim_dt)

    if recorder:
        recorder.save()
"""

import os
import torch
import cv2

# ── 녹화 설정 (AppLauncher 전에도 참조 가능) ─────────────────────────────────
# VIDEO_DIR은 LESSP_VIDEO_DIR 환경변수로 덮어쓸 수 있습니다. 기본값은 저장소 내
# videos/ 디렉토리로, .gitignore에 의해 추적되지 않습니다.
ENABLED   = True
VIDEO_DIR = os.environ.get("LESSP_VIDEO_DIR", os.path.join(os.getcwd(), "videos"))
WIDTH     = 1280
HEIGHT    = 720
FPS       = 60


def get_object_cfg():
    """씬에 추가할 DexCube RigidObjectCfg를 반환합니다. AppLauncher 이후에만 호출 가능."""
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    import isaaclab.sim as sim_utils

    return RigidObjectCfg(
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


def get_camera_cfg():
    """씬 설정에 추가할 CameraCfg를 반환합니다. AppLauncher 이후에만 호출 가능."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg

    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/RecordCamera",
        update_period=0.0,
        height=HEIGHT,
        width=WIDTH,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, f_stop=0.0, clipping_range=(0.1, 1.0e5)
        ),
    )


class SimulationRecorder:
    """시뮬레이션 영상 녹화를 관리하는 클래스.

    Args:
        video_dir: 영상 저장 디렉토리 경로.
        width: 영상 가로 해상도 (픽셀).
        height: 영상 세로 해상도 (픽셀).
        fps: 저장할 영상의 프레임레이트.
    """

    def __init__(self, video_dir: str = VIDEO_DIR, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS):
        self.video_dir = video_dir
        self.width = width
        self.height = height
        self.fps = fps
        self.frames: list = []

    def sync_camera_to_viewport(self, cam_prim_path: str = "/World/envs/env_0/RecordCamera"):
        """뷰포트 카메라 위치/방향을 RecordCamera prim에 동기화합니다.

        sim.reset() 호출 이후에 사용해야 합니다.

        Args:
            cam_prim_path: RecordCamera의 USD prim 경로.
        """
        from pxr import UsdGeom, Usd
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        vp_prim  = stage.GetPrimAtPath("/OmniverseKit_Persp")
        cam_prim = stage.GetPrimAtPath(cam_prim_path)

        if vp_prim.IsValid() and cam_prim.IsValid():
            world_mat = UsdGeom.Xformable(vp_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            xform = UsdGeom.Xformable(cam_prim)
            xform.ClearXformOpOrder()
            xform.AddTransformOp().Set(world_mat)
            print("[INFO]: RecordCamera를 뷰포트 카메라에 동기화 완료")
        else:
            print("[WARN]: 뷰포트 또는 RecordCamera prim을 찾을 수 없습니다.")

    def capture_frame(self, scene, sim_dt: float):
        """씬의 record_camera 센서에서 현재 프레임을 캡처합니다.

        시뮬레이션 루프 내 scene.update() 이후에 호출해야 합니다.

        Args:
            scene: InteractiveScene 인스턴스.
            sim_dt: 물리 시뮬레이션 타임스텝 [s].
        """
        scene.sensors["record_camera"].update(sim_dt)
        rgb_data = scene.sensors["record_camera"].data.output["rgb"]

        if rgb_data is None or rgb_data.numel() == 0:
            return

        rgb_tensor = rgb_data[0].cpu()
        if rgb_tensor.dim() == 1:
            rgb_tensor = rgb_tensor.view(self.height, self.width, -1)
        if rgb_tensor.dim() == 3 and rgb_tensor.shape[2] >= 3:
            frame = rgb_tensor[:, :, :3]
            if frame.is_floating_point():
                frame = (frame * 255).clamp(0, 255).to(torch.uint8)
            else:
                frame = frame.to(torch.uint8)
            self.frames.append(frame.clone())

    def save(self, filename: str = "simulation.mp4"):
        """캡처된 프레임을 mp4 영상으로 저장합니다.

        Args:
            filename: 저장할 파일명 (확장자 포함).
        """
        if not self.frames:
            print("[WARN]: 저장할 프레임이 없습니다.")
            return

        print(f"[INFO]: 인코딩 중... ({len(self.frames)}프레임)")
        os.makedirs(self.video_dir, exist_ok=True)
        video_path = os.path.join(self.video_dir, filename)

        h, w = self.frames[0].shape[0], self.frames[0].shape[1]
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for frame in self.frames:
            writer.write(cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR))
        writer.release()

        print(f"[INFO]: 영상 저장 완료 → {video_path}")
