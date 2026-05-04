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

try:
    import sys as _sys
    import os as _os
    _fastgs_dir = _os.path.join(_os.path.dirname(__file__), "..", "fastgs")
    if _fastgs_dir not in _sys.path:
        _sys.path.insert(0, _os.path.abspath(_fastgs_dir))
    from gaussian_renderer import render_fastgs as _render_fastgs
    from diff_gaussian_rasterization import SparseGaussianAdam as _SparseGaussianAdam
    _FASTGS_AVAILABLE = True
except ImportError:
    _render_fastgs = None
    _SparseGaussianAdam = None
    _FASTGS_AVAILABLE = False

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
        use_fast_renderer: bool = False,
        use_sparse_adam: bool = False,
    ):
        self.config = config
        self.save_results = save_results
        self.save_dir = save_dir
        self.monocular = monocular
        self.use_gui = use_gui
        self.eval_rendering = eval_rendering
        self.device = device
        self._print = print_fun or logger.info

        if use_fast_renderer and not _FASTGS_AVAILABLE:
            logger.warning(
                "use_fast_renderer=True but fastgs is not importable; "
                "falling back to the standard monogs renderer."
            )
            use_fast_renderer = False
        if use_sparse_adam and not _FASTGS_AVAILABLE:
            logger.warning(
                "use_sparse_adam=True but fastgs/SparseGaussianAdam is not "
                "importable; falling back to torch.optim.Adam."
            )
            use_sparse_adam = False
        self.use_fast_renderer = use_fast_renderer
        self.use_sparse_adam = use_sparse_adam

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

        if use_sparse_adam:
            # Replace the standard Adam with the faster sparse variant from fastgs.
            param_groups = [
                {k: v for k, v in pg.items() if k != "params"} | {"params": pg["params"]}
                for pg in self._gaussians.optimizer.param_groups
            ]
            self._gaussians.optimizer = _SparseGaussianAdam(
                param_groups, lr=0.0, eps=1e-15
            )
            self._print("GaussianSplattingManager: using SparseGaussianAdam optimizer")

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
            else:
                # Tracking: refine pose using the current gaussian map
                self._frontend.tracking(cur_frame_idx, viewpoint)

                # Keyframe decision
                last_kf_idx = self._frontend.current_window[0] if self._frontend.current_window else cur_frame_idx
                render_pkg = self._render(
                    viewpoint,
                    self._gaussians,
                    self._pipeline_params,
                    self._bg_color,
                )
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
    # Renderer dispatch
    # ------------------------------------------------------------------

    def _render(self, viewpoint, gaussians, pipeline_params, bg_color):
        """Dispatch to either the standard monogs renderer or fastgs.

        The returned dict always contains at least the keys used by the rest
        of the manager::

            render, viewspace_points, visibility_filter, radii, depth, n_touched
        """
        if not self.use_fast_renderer:
            return render(viewpoint, gaussians, pipeline_params, bg_color)

        # --- fastgs path ------------------------------------------------
        # render_fastgs signature:
        #   render_fastgs(viewpoint, pc, pipe, bg, mult, scaling_modifier,
        #                 override_color, get_flag, metric_map)
        # Returns: render, viewspace_points, visibility_filter (nonzero indices),
        #          radii, accum_metric_counts
        # Missing vs monogs: depth, opacity, n_touched
        # ----------------------------------------------------------------
        pkg = _render_fastgs(
            viewpoint, gaussians, pipeline_params, bg_color,
            mult=1.0,  # neutral multiplier
        )

        radii = pkg["radii"]

        # Build a per-Gaussian n_touched proxy: 1 where the Gaussian is visible.
        # Monogs uses n_touched > 0 only for the keyframe decision, so a binary
        # mask is sufficient.
        n_touched = (radii > 0).long()

        # fastgs visibility_filter is nonzero() indices; normalise to bool mask.
        vis_filter = pkg["visibility_filter"]
        if vis_filter.ndim > 1:
            # nonzero() returns (N,1) — convert to flat bool
            bool_mask = torch.zeros(radii.shape[0], dtype=torch.bool, device=radii.device)
            bool_mask[vis_filter.squeeze(1)] = True
            vis_filter = bool_mask

        # depth is not produced by fastgs; supply a zero tensor as a placeholder
        # so callers that check for its presence don't crash.
        depth_placeholder = torch.zeros(
            1, int(viewpoint.image_height), int(viewpoint.image_width),
            device=bg_color.device,
        )

        return {
            "render": pkg["render"],
            "viewspace_points": pkg["viewspace_points"],
            "visibility_filter": vis_filter,
            "radii": radii,
            "depth": depth_placeholder,
            "opacity": None,
            "n_touched": n_touched,
        }

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
            gaussians = self._gaussians
            xyz = gaussians.get_xyz.detach().cpu().numpy()           # (N, 3)
            features = gaussians.get_features.detach().cpu().numpy() # (N, K, 3)
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
