"""
GaussianSplattingManager
========================
A frame-by-frame online wrapper around the MonoGS SLAM system.

MonoGS was designed as a batch dataset runner; this class exposes a
live add_keyframe() API so it can be driven from an external source
(e.g. a ROS node) without any dataset on disk.

Usage::

    config = load_config("path/to/config.yaml")
    gsm = GaussianSplattingManager(config, save_dir="/tmp/gs_out",
                                   monocular=False, use_gui=False)
    gsm.start()

    for frame_id, color, depth, pose_Tcw, camera in frames:
        gsm.add_keyframe(frame_id, camera, color, depth, pose=pose_Tcw)

    points, colors = gsm.extract_point_cloud()
    gsm.save("/tmp/gs_out/final")
    gsm.stop()
"""

import os
import time
import threading
import logging

import numpy as np
import torch
import torch.multiprocessing as mp
from munch import munchify

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, focal2fov
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.camera_utils import Camera as MonoGSCamera
from utils.config_utils import load_config
from utils.multiprocessing_utils import FakeQueue, clone_obj
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd
from utils.eval_utils import save_gaussians

logger = logging.getLogger(__name__)


def _make_projection_matrix(fx, fy, cx, cy, W, H):
    return getProjectionMatrix2(
        znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
    ).transpose(0, 1)


def _color_to_tensor(color_bgr_or_rgb: np.ndarray, device="cuda:0") -> torch.Tensor:
    """Convert HxWx3 uint8/float numpy image to 3xHxW float32 cuda tensor [0,1]."""
    img = color_bgr_or_rgb.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    # assume RGB input; MonoGS expects (3, H, W)
    t = torch.from_numpy(img).permute(2, 0, 1).to(device)
    return t


def _depth_to_numpy(depth: np.ndarray) -> np.ndarray:
    """Ensure depth is a float32 HxW numpy array in metres."""
    d = depth.astype(np.float32)
    return d


def _pose_to_tensor(pose_Tcw: np.ndarray, device="cuda:0") -> torch.Tensor:
    """Convert 4x4 numpy Tcw (camera-from-world) to cuda tensor."""
    return torch.from_numpy(pose_Tcw.astype(np.float32)).to(device)


class GaussianSplattingManager:
    """Online frame-by-frame Gaussian Splatting SLAM manager.

    Parameters
    ----------
    config : dict
        MonoGS config dict (from ``load_config``).
    save_results : bool
        Whether to save the gaussian model periodically.
    save_dir : str
        Directory to save results into.
    monocular : bool
        True for monocular mode (no depth); False for RGB-D.
    live_mode : bool
        Unused; kept for API compatibility.
    use_gui : bool
        Whether to launch the 3D visualisation GUI.
    eval_rendering : bool
        Whether to run rendering evaluation on finish.
    use_dataset : bool
        Unused; kept for API compatibility.
    print_fun : callable
        Logging callback (default: logger.info).
    device : str
        Torch device string.
    """

    def __init__(
        self,
        config: dict,
        save_results: bool = True,
        save_dir: str = "results/gaussian_splatting",
        monocular: bool = False,
        live_mode: bool = False,
        use_gui: bool = False,
        eval_rendering: bool = False,
        use_dataset: bool = False,
        print_fun=None,
        device: str = "cuda:0",
    ):
        self.config = config
        self.save_results = save_results
        self.save_dir = save_dir
        self.monocular = monocular
        self.use_gui = use_gui
        self.eval_rendering = eval_rendering
        self.device = device
        self._print = print_fun or logger.info

        # Patch config for live use
        self.config["Results"]["save_results"] = save_results
        self.config["Results"]["save_dir"] = save_dir
        self.config["Results"]["use_gui"] = use_gui
        self.config["Results"]["eval_rendering"] = False  # disabled in online mode
        self.config["Training"]["monocular"] = monocular
        # Live mode: treat as realsense-style so frontend doesn't look for dataset
        self.config["Dataset"]["type"] = "realsense"
        self.config["Dataset"]["sensor_type"] = "monocular" if monocular else "rgbd"

        self._model_params = munchify(config["model_params"])
        self._opt_params = munchify(config["opt_params"])
        self._pipeline_params = munchify(config["pipeline_params"])

        self._bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        self._gaussians = GaussianModel(self._model_params.sh_degree, config=config)
        self._gaussians.init_lr(6.0)
        self._gaussians.training_setup(self._opt_params)

        self._frontend_queue = mp.Queue()
        self._backend_queue = mp.Queue()
        self._q_main2vis = mp.Queue() if use_gui else FakeQueue()
        self._q_vis2main = mp.Queue() if use_gui else FakeQueue()

        self._backend = BackEnd(config)
        self._backend.gaussians = self._gaussians
        self._backend.background = self._bg_color
        self._backend.cameras_extent = 6.0
        self._backend.pipeline_params = self._pipeline_params
        self._backend.opt_params = self._opt_params
        self._backend.frontend_queue = self._frontend_queue
        self._backend.backend_queue = self._backend_queue
        self._backend.live_mode = True
        self._backend.set_hyperparams()

        # Frontend used only for its tracking/keyframe helpers — we don't call run()
        self._frontend = FrontEnd(config)
        self._frontend.gaussians = self._gaussians
        self._frontend.background = self._bg_color
        self._frontend.pipeline_params = self._pipeline_params
        self._frontend.frontend_queue = self._frontend_queue
        self._frontend.backend_queue = self._backend_queue
        self._frontend.q_main2vis = self._q_main2vis
        self._frontend.q_vis2main = self._q_vis2main
        self._frontend.set_hyperparams()

        self._backend_process: mp.Process | None = None
        self._gui_process: mp.Process | None = None

        # Per-camera projection matrix cache (keyed by (fx,fy,cx,cy,W,H))
        self._proj_matrix_cache: dict = {}

        self._frame_idx = 0
        self._lock = threading.Lock()
        self._started = False

        mkdir_p(save_dir)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Launch the backend (and optional GUI) processes."""
        if self._started:
            return
        self._backend_process = mp.Process(target=self._backend.run)
        self._backend_process.start()
        if self.use_gui:
            from gui import slam_gui, gui_utils
            params_gui = gui_utils.ParamsGUI(
                pipe=self._pipeline_params,
                background=self._bg_color,
                gaussians=self._gaussians,
                q_main2vis=self._q_main2vis,
                q_vis2main=self._q_vis2main,
            )
            self._gui_process = mp.Process(target=slam_gui.run, args=(params_gui,))
            self._gui_process.start()
            time.sleep(3)
        self._started = True
        self._print("GaussianSplattingManager: started")

    def stop(self):
        """Signal the backend to stop and join all processes."""
        if not self._started:
            return
        self._backend_queue.put(["stop"])
        if self._backend_process is not None:
            self._backend_process.join(timeout=30)
        if self._gui_process is not None:
            self._q_main2vis.put({"finish": True})
            self._gui_process.join(timeout=10)
        self._started = False
        self._print("GaussianSplattingManager: stopped")

    def reset(self):
        """Reset the SLAM state (clear gaussians and frontend state)."""
        with self._lock:
            self._backend_queue.put(["reset"])
            self._frontend.reset = True
            self._frontend.initialized = False
            self._frontend.cameras.clear()
            self._frame_idx = 0
        self._print("GaussianSplattingManager: reset")

    def _drain_frontend_queue(self, block=False, timeout=0.5):
        """Drain pending messages from the backend->frontend queue.

        When block=True, waits up to *timeout* seconds for at least one message
        (used after sending 'init' to wait for the backend to respond).
        """
        drained = 0
        if block:
            try:
                data = self._frontend_queue.get(timeout=timeout)
                self._frontend.sync_backend(data)
                if data[0] == "init":
                    self._frontend.requested_init = False
                elif data[0] == "keyframe":
                    self._frontend.requested_keyframe = max(
                        0, self._frontend.requested_keyframe - 1
                    )
                drained += 1
            except Exception:
                pass
        while not self._frontend_queue.empty():
            try:
                data = self._frontend_queue.get_nowait()
                self._frontend.sync_backend(data)
                if data[0] == "init":
                    self._frontend.requested_init = False
                elif data[0] == "keyframe":
                    self._frontend.requested_keyframe = max(
                        0, self._frontend.requested_keyframe - 1
                    )
                drained += 1
            except Exception:
                break
        return drained

    # ------------------------------------------------------------------
    # Core frame-by-frame API
    # ------------------------------------------------------------------

    def add_keyframe(
        self,
        frame_id: int,
        camera,
        color: np.ndarray,
        depth: np.ndarray | None,
        pose: np.ndarray | None = None,
        gt_pose: np.ndarray | None = None,
        feature_distillation_payload: dict | None = None,
    ):
        """Add a single keyframe to the SLAM system.

        Parameters
        ----------
        frame_id : int
            Unique identifier for this frame.
        camera :
            gs_slam Camera object with .fx, .fy, .cx, .cy, .width, .height.
        color : np.ndarray
            RGB image, HxWx3, uint8 or float32 [0,1].
        depth : np.ndarray | None
            Depth image HxW in metres. None for monocular.
        pose : np.ndarray | None
            4x4 Tcw pose (camera-from-world). Identity if None.
        gt_pose : np.ndarray | None
            4x4 ground-truth Tcw pose for evaluation. Same as pose if None.
        feature_distillation_payload : dict | None
            Optional distillation features (ignored if backend doesn't support).
        """
        with self._lock:
            proj_key = (camera.fx, camera.fy, camera.cx, camera.cy,
                        camera.width, camera.height)
            if proj_key not in self._proj_matrix_cache:
                self._proj_matrix_cache[proj_key] = _make_projection_matrix(
                    camera.fx, camera.fy, camera.cx, camera.cy,
                    camera.width, camera.height
                )
            proj_matrix = self._proj_matrix_cache[proj_key]

            color_t = _color_to_tensor(color, device=self.device)
            depth_np = _depth_to_numpy(depth) if depth is not None else None

            if pose is None:
                pose_np = np.eye(4, dtype=np.float32)
            else:
                pose_np = pose.astype(np.float32)
            if gt_pose is None:
                gt_pose_np = pose_np
            else:
                gt_pose_np = gt_pose.astype(np.float32)

            gt_T = _pose_to_tensor(gt_pose_np, device=self.device)

            fovx = focal2fov(camera.fx, camera.width)
            fovy = focal2fov(camera.fy, camera.height)

            viewpoint = MonoGSCamera(
                uid=frame_id,
                color=color_t,
                depth=depth_np,
                gt_T=gt_T,
                projection_matrix=proj_matrix,
                fx=camera.fx,
                fy=camera.fy,
                cx=camera.cx,
                cy=camera.cy,
                fovx=fovx,
                fovy=fovy,
                image_height=camera.height,
                image_width=camera.width,
                device=self.device,
            )
            viewpoint.compute_grad_mask(self.config)

            # Set the estimated pose
            pose_t = _pose_to_tensor(pose_np, device=self.device)
            viewpoint.update_RT(pose_t[:3, :3], pose_t[:3, 3])

            self._frontend.cameras[frame_id] = viewpoint
            cur_frame_idx = frame_id

            if self._frontend.reset or not self._frontend.initialized:
                self._frontend.initialize(cur_frame_idx, viewpoint)
                self._frontend.current_window.append(cur_frame_idx)
                # Block-wait for the backend to respond with initial Gaussians
                self._drain_frontend_queue(block=True, timeout=5.0)
            else:
                # Drain any pending backend updates before tracking/rendering
                self._drain_frontend_queue(block=False)
                # Tracking: refine pose using the current gaussian map
                self._frontend.tracking(cur_frame_idx, viewpoint)

                # Keyframe decision
                last_kf_idx = self._frontend.current_window[0] if self._frontend.current_window else cur_frame_idx
                render_pkg = render(
                    viewpoint,
                    self._frontend.gaussians,
                    self._pipeline_params,
                    self._bg_color,
                )
                if render_pkg is None:
                    # Gaussians not yet initialized; skip keyframe decision
                    self._frame_idx += 1
                    return
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                create_kf = self._frontend.is_keyframe(
                    cur_frame_idx,
                    last_kf_idx,
                    curr_visibility,
                    self._frontend.occ_aware_visibility,
                )

                if create_kf:
                    self._frontend.request_keyframe(
                        cur_frame_idx, viewpoint,
                        self._frontend.current_window,
                        render_pkg.get("depth"),
                    )

            self._frame_idx += 1

    # ------------------------------------------------------------------
    # Output / persistence
    # ------------------------------------------------------------------

    def extract_point_cloud(self):
        """Extract a rough point cloud from the current Gaussian map.

        Returns
        -------
        points : np.ndarray | None  shape (N, 3)
        colors : np.ndarray | None  shape (N, 3)
        """
        try:
            # Use the frontend's gaussians — these are kept up-to-date via
            # sync_backend() which is called from _drain_frontend_queue().
            gaussians = self._frontend.gaussians
            if gaussians is None:
                return None, None
            xyz = gaussians.get_xyz.detach().cpu().numpy()           # (N, 3)
            features = gaussians.get_features.detach().cpu().numpy() # (N, K, 3)
            if xyz.ndim != 2 or xyz.shape[1] != 3 or features.ndim != 3:
                return None, None
            # Use the DC (zeroth-order) SH feature as colour
            colors = features[:, 0, :]   # (N, 3)
            colors = (colors * 0.28209479177387814 + 0.5).clip(0, 1)
            return xyz, colors
        except Exception as e:
            self._print(f"GaussianSplattingManager.extract_point_cloud: {e}")
            return None, None

    def save(self, path: str):
        """Save the Gaussian model to *path*."""
        try:
            mkdir_p(path)
            save_gaussians(self._gaussians, path, "final", final=True)
            self._print(f"GaussianSplattingManager: saved to {path}")
        except Exception as e:
            self._print(f"GaussianSplattingManager.save: {e}")
