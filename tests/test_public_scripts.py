import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_cmd(args, cwd=ROOT, check=False):
    return subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)


def make_minimal_occscannet(root: Path):
    data = root / "OccScanNet"
    scene = "scene0000_00"
    token = f"{scene}/00000"
    (data / "gathered_data" / scene).mkdir(parents=True)
    (data / "posed_images" / scene).mkdir(parents=True)
    (data / "gts_camvisbits" / scene / "00000").mkdir(parents=True)
    gathered = {
        "target_1_4": np.zeros((60, 60, 36), dtype=np.uint8),
        "voxel_origin": np.array([0.0, 0.0, 0.0], dtype=np.float32),
        "cam_pose": np.eye(4, dtype=np.float64),
        "intrinsic": np.eye(4, dtype=np.float64),
    }
    with (data / "gathered_data" / scene / "00000.pkl").open("wb") as f:
        pickle.dump(gathered, f)
    (data / "posed_images" / scene / "00000.jpg").write_bytes(b"fake-jpeg-placeholder")
    label = data / "gts_camvisbits" / scene / "00000" / "labels.npz"
    np.savez_compressed(
        label,
        semantics=np.zeros((130, 120, 140), dtype=np.uint8),
        mask_lidar=np.ones((130, 120, 140), dtype=np.uint8),
        mask_camera=np.ones((130, 120, 140), dtype=np.uint8),
        raw_semantics=np.zeros((60, 60, 36), dtype=np.uint8),
        voxel_origin=np.zeros(3, dtype=np.float32),
        voxel_size=np.array([0.08, 0.08, 0.08], dtype=np.float32),
    )
    info = {
        "token": token,
        "scene_name": scene,
        "lidar_path": f"gathered_data/{scene}/00000.pkl",
        "ego2global_rotation": np.eye(3, dtype=np.float32),
        "ego2global_translation": np.zeros(3, dtype=np.float32),
        "cams": {
            "CAM_FRONT": {
                "data_path": f"posed_images/{scene}/00000.jpg",
                "depth_path": f"depth_splatssc_stage1_ftdav2_vitb_20m_full/{scene}/00000.png",
                "cam_intrinsic": np.eye(3, dtype=np.float32),
                "sensor2lidar_rotation": np.eye(3, dtype=np.float32),
                "sensor2lidar_translation": np.zeros(3, dtype=np.float32),
            }
        },
    }
    for name in [
        "train_occscannet_mini.pkl",
        "val_occscannet_mini.pkl",
        "test_occscannet_mini.pkl",
    ]:
        with (data / name).open("wb") as f:
            pickle.dump({"metadata": {}, "infos": [info]}, f)
    return data


def test_check_assets_reports_missing_and_passes_online_without_precomputed_depth(tmp_path):
    data = make_minimal_occscannet(tmp_path)
    pretrain = tmp_path / "pretrain"
    (pretrain / "depth_anything").mkdir(parents=True)
    (pretrain / "fusion_pretrain_model.pth").write_bytes(b"stub")
    (pretrain / "depth_anything" / "finetune_scannet_depthanythingv2.pth").write_bytes(b"stub")
    result = run_cmd([
        PYTHON,
        "scripts/check_assets.py",
        "--data-root", str(data),
        "--pretrain-root", str(pretrain),
        "--online-depth",
        "--json",
    ])
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["counts"]["infos"] == 3
    assert not payload["missing"]

    # Precomputed depth is optional for online-depth, but required when explicitly requested.
    result = run_cmd([
        PYTHON,
        "scripts/check_assets.py",
        "--data-root", str(data),
        "--pretrain-root", str(pretrain),
        "--precomputed-depth",
        "--json",
    ])
    assert result.returncode != 0
    payload = json.loads(result.stdout)
    assert payload["ok"] is False
    assert any(item["kind"] == "depth" for item in payload["missing"])


def test_check_loss_scale_detects_query_and_loss_window(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("""
06/02 00:00 train_runtime_num_query: 100.0000 loss: 6.2 loss_containment: 0.012
06/02 00:01 loss: 5.8 loss_containment: 0.011
""")
    result = run_cmd([PYTHON, "scripts/check_loss_scale.py", str(log), "--first-n", "2", "--json"])
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["num_query_seen"] == 100
    assert payload["loss_count"] == 2

    bad = tmp_path / "bad.log"
    bad.write_text("train_runtime_num_query: 100.0000 loss: nan loss_containment: 0.0\n")
    result = run_cmd([PYTHON, "scripts/check_loss_scale.py", str(bad), "--json"])
    assert result.returncode != 0


def test_generate_occscannet_mini_pkls_matches_expected_transform(tmp_path):
    data = make_minimal_occscannet(tmp_path)
    (data / "train_subscenes.txt").write_text("gathered_data/scene0000_00/00000.pkl\n")
    (data / "val_subscenes.txt").write_text("gathered_data/scene0000_00/00000.pkl\n")
    result = run_cmd([
        PYTHON,
        "scripts/generate_occscannet_mini_pkls.py",
        "--data-root", str(data),
        "--overwrite",
    ])
    assert result.returncode == 0, result.stderr + result.stdout
    with (data / "train_occscannet_mini.pkl").open("rb") as f:
        obj = pickle.load(f)
    info = obj["infos"][0]
    assert info["token"] == "scene0000_00/00000"
    assert info["cams"]["CAM_FRONT"]["data_path"] == "posed_images/scene0000_00/00000.jpg"
    assert info["cams"]["CAM_FRONT"]["depth_path"].endswith("scene0000_00/00000.png")
    assert np.asarray(info["cams"]["CAM_FRONT"]["cam_intrinsic"]).shape == (4, 4)
    assert np.allclose(info["cams"]["CAM_FRONT"]["sensor2lidar_rotation"], [[1,0,0],[0,0,1],[0,-1,0]])



def test_online_depth_points_projects_and_freezes_fake_model():
    import pytest

    torch = pytest.importorskip("torch")
    from models.adaocc.online_depth import OnlineDepthAnythingPoints

    class FakeDepth(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

        def forward(self, image, output_feature=False):
            # Return a 2m dense depth map at the same H/W as input.
            return torch.full((image.shape[0], image.shape[-2], image.shape[-1]), 2.0, device=image.device)

    module = OnlineDepthAnythingPoints(
        enabled=True,
        depth_model=FakeDepth(),
        input_size=None,
        depth_min=0.1,
        depth_max=7.5,
        sample_stride=1,
        image_is_bgr=False,
    )
    assert module.depth_model.training is False
    assert all(not p.requires_grad for p in module.depth_model.parameters())

    image = np.zeros((2, 2, 3), dtype=np.uint8)
    sample = {
        "timestamp": 0.0,
        "views": [{
            "image": image,
            "view_idx": 0,
            "timestamp": 0.0,
            "cam_intrinsic": np.eye(3, dtype=np.float32),
            "sensor2lidar_rotation": np.eye(3, dtype=np.float32),
            "sensor2lidar_translation": np.zeros(3, dtype=np.float32),
        }],
    }
    points, view_ids = module([sample], device=torch.device("cpu"), batch_size=1)
    assert len(points) == 1
    assert len(view_ids) == 1
    assert points[0].shape == (4, 5)
    assert torch.all(view_ids[0] == 0)
    # Pixel-center projection with identity intrinsics: x=(u+0.5)*z, y=(v+0.5)*z, z=2.
    expected_xyz = torch.tensor([
        [1.0, 1.0, 2.0],
        [3.0, 1.0, 2.0],
        [1.0, 3.0, 2.0],
        [3.0, 3.0, 2.0],
    ])
    assert torch.allclose(points[0][:, :3], expected_xyz)
    assert torch.allclose(points[0][:, 3:], torch.zeros((4, 2)))


def test_local_paths_template_and_wrappers_are_config_file_driven():
    template = ROOT / "configs" / "local_paths.example.sh"
    assert template.exists()
    text = template.read_text()
    assert "Copy this file to configs/local_paths.sh" in text
    assert "ADAOCC_DATA_ROOT" in text
    assert "ADAOCC_ONLINE_DEPTH" in text
    assert "ADAOCC_DISABLE_MSMV_CUDA" in text

    for rel in ["dist_train.sh", "dist_val.sh", "scripts/link_local_assets.sh"]:
        script = (ROOT / rel).read_text()
        assert "configs/local_paths.sh" in script
        assert "ADAOCC_LOCAL_CONFIG" in script

def test_msmv_fallback_disable_suppresses_optional_extension_warning():
    import pytest

    pytest.importorskip("torch")
    env = os.environ.copy()
    env["ADAOCC_DISABLE_MSMV_CUDA"] = "1"
    result = subprocess.run(
        [PYTHON, "-c", "import models.csrc.wrapper as w; print(w.MSMV_CUDA)"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    combined = result.stdout + result.stderr
    assert "_msmv_sampling_cuda" not in combined
    assert "failed to load" not in combined
    assert "False" in result.stdout

