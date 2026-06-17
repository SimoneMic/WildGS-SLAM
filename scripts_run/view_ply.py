import argparse
import torch
import open3d as o3d
from munch import munchify

from src import config
from src.gui import gui_utils, slam_gui
from thirdparty.gaussian_splatting.scene.gaussian_model import GaussianModel

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file used for the run.')
    parser.add_argument('ply', type=str, help='Path to the saved final_gs.ply to visualize.')
    args = parser.parse_args()

    cfg = config.load_config(args.config)

    pipeline_params = munchify(cfg['mapping']['pipeline_params'])
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')

    gaussians = GaussianModel(cfg['mapping']['model_params']['sh_degree'], config=cfg)
    gaussians.load_ply(args.ply)

    params_gui = gui_utils.ParamsGUI(
        pipe=pipeline_params,
        background=background,
        gaussians=gaussians,
        q_main2vis=None,
        q_vis2main=None,
    )

    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = slam_gui.SLAM_GUI(params_gui)

    # No live tracking frames are sent in offline mode, so the 3D widget's
    # camera is never aimed at the scene by look_at(); point it at the
    # loaded Gaussians' bounding box ourselves.
    pts = gaussians.get_xyz.detach().cpu().numpy()
    aabb = o3d.geometry.AxisAlignedBoundingBox(pts.min(axis=0), pts.max(axis=0))
    win.widget3d.setup_camera(60.0, aabb, aabb.get_center())

    app.run()
