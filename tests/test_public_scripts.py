import json
import os
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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


def test_check_assets_reports_missing_and_passes_online_raw_without_precomputed_depth(tmp_path):
    data = make_minimal_occscannet(tmp_path)
    pretrain = tmp_path / "pretrain"
    (pretrain / "depth_anything").mkdir(parents=True)
    (pretrain / "radio" / "C-RADIOv3-B").mkdir(parents=True)
    (pretrain / "fusion_pretrain_model.pth").write_bytes(b"stub")
    (pretrain / "depth_anything" / "finetune_scannet_depthanythingv2.pth").write_bytes(b"stub")
    (pretrain / "radio" / "C-RADIOv3-B" / "config.json").write_text("{}")
    import pytest

    Image = pytest.importorskip("PIL.Image")
    Image.fromarray(np.array([[1000, 2000], [0, 3000]], dtype=np.uint16)).save(
        data / "posed_images" / "scene0000_00" / "00000.png"
    )

    def run_asset_check(*extra_args):
        command = [
            PYTHON,
            "scripts/check_assets.py",
            "--data-root", str(data),
            "--pretrain-root", str(pretrain),
        ]
        command.extend(extra_args)
        command.append("--json")
        result = run_cmd(command)
        return result, json.loads(result.stdout)

    result, payload = run_asset_check("--online-depth")
    assert result.returncode == 0, result.stderr + result.stdout
    assert payload["ok"] is True
    assert payload["mode"]["radio"] is True
    assert payload["counts"]["infos"] == 3
    assert not payload["missing"]

    result, payload = run_asset_check("--raw-depth-from-images", "--verify-depth-png")
    assert result.returncode == 0, result.stderr + result.stdout
    assert payload["ok"] is True
    assert payload["mode"]["raw_depth_from_images"] is True
    assert payload["counts"]["depth_paths_checked"] == 3
    assert payload["counts"]["depth_decoded"] == 3
    assert payload["samples"]["first"]["raw_depth"].endswith("posed_images/scene0000_00/00000.png")
    assert not payload["missing"]

    shutil.rmtree(pretrain / "radio")
    result, payload = run_asset_check()
    assert result.returncode != 0
    assert payload["mode"]["radio"] is True
    assert any(item["kind"] == "radio" for item in payload["missing"])

    result, payload = run_asset_check("--radio")
    assert result.returncode != 0
    assert payload["mode"]["radio"] is True
    assert any(item["kind"] == "radio" for item in payload["missing"])

    result, payload = run_asset_check("--efficientnet-b7")
    assert result.returncode != 0
    assert payload["mode"]["radio"] is False
    assert payload["mode"]["efficientnet_b7"] is True
    assert any(item["kind"] == "efficientnet_b7_checkpoint" for item in payload["missing"])
    assert any(item["path"].endswith("pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth") for item in payload["missing"])
    assert not any(item["kind"] == "radio" for item in payload["missing"])
    assert all(key != "checkpoints" + "_root" for key in payload)

    (pretrain / "timm").mkdir()
    (pretrain / "timm" / "tf_efficientnet_b7_ns-1dbc32de.pth").write_bytes(b"stub")
    result, payload = run_asset_check("--efficientnet-b7")
    assert result.returncode == 0, result.stderr + result.stdout
    assert payload["ok"] is True
    assert payload["mode"]["radio"] is False
    assert payload["mode"]["efficientnet_b7"] is True
    assert not payload["missing"]

    # Precomputed depth is optional for online-depth, but required when explicitly requested.
    result, payload = run_asset_check("--precomputed-depth")
    assert result.returncode != 0
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
    pytest.importorskip("mmengine")
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


def test_public_wrappers_use_fixed_repo_relative_layout():
    assert not (ROOT / "configs" / "local_paths.example.sh").exists()
    assert not (ROOT / "configs" / "current.py").exists()
    assert not (ROOT / "configs" / "shared" / "runtime.py").exists()
    assert not (ROOT / "scripts" / "link_local_assets.sh").exists()

    for rel in ["dist_train.sh", "dist_val.sh"]:
        script = (ROOT / rel).read_text()
        assert "configs/local_paths.sh" not in script
        assert "ADAOCC_LOCAL_CONFIG" not in script
        assert "export ADAOCC_" not in script
        assert "export HF_HOME" not in script
        assert "CUDA_VISIBLE_DEVICES" not in script
        assert "NCCL_" not in script
        assert "torch.distributed.run" in script
        assert "PYTHON=${PYTHON:-python}" not in script

    train_script = (ROOT / "dist_train.sh").read_text()
    assert 'GPUS=${1:-8}' in train_script
    assert 'CONFIG=${2:-configs/occscannet/radio_occscannet_mini.py}' in train_script
    assert '--config "$CONFIG"' in train_script
    assert '"$@"' in train_script

    val_script = (ROOT / "dist_val.sh").read_text()
    assert 'GPUS=${1:-8}' in val_script
    assert 'CONFIG=${2:-configs/occscannet/radio_occscannet_mini.py}' in val_script
    assert 'WEIGHT=${3:?usage: dist_val.sh GPUS CONFIG WEIGHT [MASTER_PORT]}' in val_script

    train_py = (ROOT / "train.py").read_text()
    assert "--run-label" in train_py
    assert "--output-root" in train_py
    assert "--work-dir" in train_py

    cfg = (ROOT / "configs" / "occscannet" / "radio_occscannet_mini.py").read_text()
    assert 'dataset_root = str(_repo_root / "data" / "OccScanNet")' in cfg
    assert 'train_ann_file = str(_Path(dataset_root) / "train_occscannet_mini.pkl")' in cfg
    assert 'radio_model_id = str(_repo_root / "pretrain" / "radio" / "C-RADIOv3-B")' in cfg
    assert 'custom_imports = dict(imports=["models", "loaders"], allow_failed_imports=False)' in cfg
    assert '_base_' not in cfg


def test_public_docs_describe_config_choice_and_depth_defaults():
    readme = (ROOT / "README.md").read_text()
    data_doc = (ROOT / "docs" / "DATA.md").read_text()
    ai_doc = (ROOT / "docs" / "AI_REPRODUCTION.md").read_text()
    repro = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text()
    arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text()
    dep = (ROOT / "docs" / "DEPENDENCY_TRACE.md").read_text()
    license_doc = (ROOT / "docs" / "LICENSE_AND_ASSETS.md").read_text()

    assert "docs/AI_REPRODUCTION.md" in readme
    assert "AI-agent AdaOcc reproduction guide" in ai_doc
    assert "do not modify unrelated model, data-loader, training, evaluation, or config logic" in ai_doc
    assert "do not commit or push unless explicitly asked" in ai_doc
    assert "Do not use global `CONFIG` or `SMOKE_CONFIG`" in ai_doc
    assert "Choose config, depth mode, and run" in readme
    assert "Choose image encoder by config path" in readme
    assert "### 5.4 Final expected layout" in readme
    assert "### 5.5 Smoke, train, and eval examples" in readme
    assert readme.index("### 5.4 Final expected layout") < readme.index("### 5.5 Smoke, train, and eval examples")
    assert "data/OccScanNet/" in readme
    assert "train_occscannet_mini.pkl        # generated PKLs" in readme
    assert "pretrain/                                         # external/pretrained weights/initializers" in readme
    assert "gts_camvisbits/<scene>/<frame>/labels.npz" in readme
    assert "posed_images/<scene>/<frame>.{jpg,png}" in readme
    assert "depth_splatssc_stage1_ftdav2_vitb_20m_full/<scene>/<frame>.png" in readme
    assert "optional generated precomputed depth" in readme
    assert "checkpoints/                                      # trained AdaOcc model checkpoints" in readme
    assert "adaocc_online_depth_occscannet_mini_epoch200.pth" in readme

    for text in (readme, ai_doc, repro, arch, dep):
        assert "configs/occscannet/radio_occscannet_mini.py" in text
        assert "configs/occscannet/efficientnet_b7_occscannet_mini.py" in text
    for text in (readme, data_doc, ai_doc, repro, arch):
        assert "posed_images/<scene>/<frame>.png" in text
        assert "ADAOCC_ONLINE_DEPTH=1" in text
        assert "ADAOCC_RAW_DEPTH_FROM_IMAGES=0" in text
        assert "depth_splatssc_stage1_ftdav2_vitb_20m_full" in text
        assert "pretrain/depth_anything/finetune_scannet_depthanythingv2.pth" in text

    assert "RADIO remains the released/reference baseline" in readme
    assert "EfficientNet-B7 additional option" in readme
    assert "pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in readme
    assert "wget -O pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in readme
    assert "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth" in readme
    assert "pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in data_doc
    assert "pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in ai_doc
    assert "pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in repro
    assert "pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in arch
    assert "pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in dep
    assert "pretrain/timm/tf_efficientnet_b7_ns-1dbc32de.pth" in license_doc
    assert "tf_efficientnet_b7_ns-1dbc32de.pth" in readme
    assert "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_efficientnet_b7_ns-1dbc32de.pth" in license_doc
    assert "tf_efficientnet_b7_ns-1dbc32de.pth" in license_doc
    assert "--efficientnet-b7" in readme
    assert "--radio" in readme
    assert "--radio" in ai_doc
    assert "--radio" in repro
    assert "--efficientnet-b7" in data_doc
    assert "--efficientnet-b7" in ai_doc
    assert "--efficientnet-b7" in dep
    assert "Default depth mode: local/prepared" in repro
    for text in (readme, ai_doc, repro):
        assert "CONFIG=configs/" not in text
        assert "SMOKE_CONFIG=" not in text
        assert "$CONFIG" not in text
        assert "$SMOKE_CONFIG" not in text
        assert "ADAOCC_DISABLE_MSMV_CUDA" in text
        assert "PyTorch fallback" in text
        assert text.count("ADAOCC_DISABLE_MSMV_CUDA=1") == 1
        assert "ADAOCC_ONLINE_DEPTH=1 ADAOCC_DISABLE_MSMV_CUDA=1" not in text
        assert "ADAOCC_RAW_DEPTH_FROM_IMAGES=0 ADAOCC_DISABLE_MSMV_CUDA=1" not in text
        assert "./dist_train.sh 8 configs/occscannet/radio_occscannet_mini_smoke.py" in text
        assert "./dist_train.sh 8 configs/occscannet/efficientnet_b7_occscannet_mini_smoke.py" in text
    for text in (readme, ai_doc, repro):
        assert "ADAOCC_ONLINE_DEPTH=1" in text
        assert "ADAOCC_RAW_DEPTH_FROM_IMAGES=0" in text
        assert "local/prepared" in text
        assert "./dist_val.sh 8 configs/occscannet/radio_occscannet_mini.py" in text
        assert "checkpoints/adaocc_online_depth_occscannet_mini_epoch200.pth" in text
        assert "released checkpoint" in text
        assert "Default local/prepared" in text
    assert "selected frozen image backbone" not in arch
    assert "unfreeze_last_n_blocks=4" in arch
    assert "EfficientNet-B7 backbone weights are frozen" in arch
    assert "Default public reproduction uses online DepthAnything" not in data_doc
    assert "Default depth mode: online DepthAnything" not in repro
    assert "Use precomputed-depth mode by setting `ADAOCC_ONLINE_DEPTH=0`" not in data_doc


def test_msmv_fallback_disable_suppresses_optional_extension_warning():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("mmengine")
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


def test_extract_fusion_pretrain_filters_only_current_adaocc_middle_encoder():
    from collections import OrderedDict
    from scripts.extract_adaocc_fusion_pretrain import DEFAULT_PREFIXES, filter_state_dict

    state = OrderedDict([
        ("img_backbone.patch_embed.proj.weight", object()),
        ("pts_middle_encoder.conv_input.0.weight", object()),
        ("pts_backbone.blocks.0.weight", object()),
        ("pts_neck.lateral_convs.0.weight", object()),
        ("occ_head.weight", object()),
    ])
    selected = filter_state_dict(state)

    assert DEFAULT_PREFIXES == ("pts_middle_encoder.",)
    assert list(selected) == ["pts_middle_encoder.conv_input.0.weight"]


def test_extract_fusion_pretrain_converts_5d_spconv_kernel_layout():
    from collections import OrderedDict
    from scripts.extract_adaocc_fusion_pretrain import convert_spconv_kernels_for_load_hook

    class FakeKernel:
        ndim = 5

        def permute(self, *dims):
            self.dims = dims
            return self

        def contiguous(self):
            return ("converted", self.dims)

    kernel = FakeKernel()
    state = OrderedDict([
        ("pts_middle_encoder.conv_input.0.weight", kernel),
        ("pts_middle_encoder.conv_input.1.weight", object()),
    ])

    converted, keys = convert_spconv_kernels_for_load_hook(state)

    assert keys == ["pts_middle_encoder.conv_input.0.weight"]
    assert converted["pts_middle_encoder.conv_input.0.weight"] == ("converted", (1, 2, 3, 4, 0))
    assert converted["pts_middle_encoder.conv_input.1.weight"] is state["pts_middle_encoder.conv_input.1.weight"]


def test_efficientnet_config_and_lazy_timm_contracts():
    eff_cfg = ROOT / "configs" / "occscannet" / "efficientnet_b7_occscannet_mini.py"
    smoke_cfg = ROOT / "configs" / "occscannet" / "efficientnet_b7_occscannet_mini_smoke.py"
    radio_cfg = ROOT / "configs" / "occscannet" / "radio_occscannet_mini.py"
    wrapper = ROOT / "models" / "backbones" / "timm_feature_backbone.py"
    init_file = ROOT / "models" / "backbones" / "__init__.py"

    eff_text = eff_cfg.read_text()
    smoke_text = smoke_cfg.read_text()
    radio_text = radio_cfg.read_text()
    wrapper_text = wrapper.read_text()
    init_text = init_file.read_text()

    assert "models.backbones.timm_feature_backbone" in eff_text
    assert "models.backbones.timm_feature_backbone" not in radio_text
    assert "_base_" not in eff_text
    assert "TimmFeatureBackbone" in eff_text
    assert 'pretrain" / "timm" / "tf_efficientnet_b7_ns-1dbc32de.pth"' in eff_text
    assert "tf_efficientnet_b7_ns-1dbc32de.pth" in eff_text
    assert 'norm_type="imagenet"' in eff_text
    assert "patch_size=16" in eff_text
    assert "expected_output_stride=image_feature_output_divisor" in eff_text

    assert "import timm" not in "\n".join(wrapper_text.splitlines()[:20])
    assert "'TimmFeatureBackbone'" in init_text
    assert "_LAZY_ONLY_BACKBONES" in init_text
    assert "if name not in _LAZY_ONLY_BACKBONES" in init_text

    assert "global_batch_size = 1" in smoke_text
    assert "batch_size = 1" in smoke_text
    assert 'type="IterBasedTrainLoop"' in smoke_text
    assert "max_iters=1" in smoke_text
    assert "val_cfg = None" in smoke_text
    assert "val_dataloader = None" in smoke_text
    assert "val_evaluator = None" in smoke_text
    assert "_delete_=True" in smoke_text
    assert 'type="DefaultSampler"' in smoke_text



def test_efficientnet_checkpoint_prefix_policy_covers_local_checkpoint():
    import pytest

    torch = pytest.importorskip("torch")
    checkpoint = ROOT / "pretrain" / "timm" / "tf_efficientnet_b7_ns-1dbc32de.pth"
    if not checkpoint.exists():
        pytest.skip("EfficientNet checkpoint is not available")

    state_dict = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state_dict, dict) and not any(str(k).startswith("conv_stem") for k in state_dict):
        for key in ("state_dict", "model"):
            if isinstance(state_dict.get(key), dict):
                state_dict = state_dict[key]
                break
    prefixes = {str(key).split(".", 1)[0] for key in state_dict}
    trunk_prefixes = {"conv_stem", "bn1", "blocks"}
    post_feature_prefixes = prefixes - trunk_prefixes

    wrapper_text = (ROOT / "models" / "backbones" / "timm_feature_backbone.py").read_text()
    allowed_prefixes = {
        item.strip().rstrip(",").strip("\"'").rstrip(".")
        for item in wrapper_text.split("_ALLOWED_UNEXPECTED_PREFIXES = (", 1)[1]
        .split(")", 1)[0]
        .splitlines()
        if item.strip().startswith(("\"", "'"))
    }

    assert trunk_prefixes.issubset(prefixes)
    assert post_feature_prefixes.issubset(allowed_prefixes)


def test_full_split_config_and_checkpoint_checker_contracts():
    full_cfg = ROOT / "configs" / "occscannet" / "radio_occscannet_full.py"
    checker = ROOT / "scripts" / "check_checkpoint.py"
    assert full_cfg.is_file()
    assert checker.is_file()

    full_text = full_cfg.read_text()
    assert "_base_" not in full_text
    assert 'train_ann_file = str(_Path(dataset_root) / "train_occscannet_full.pkl")' in full_text
    assert 'val_ann_file = str(_Path(dataset_root) / "val_occscannet_full.pkl")' in full_text
    assert 'test_ann_file = str(_Path(dataset_root) / "test_occscannet_full.pkl")' in full_text
    assert "total_epochs = 100" in full_text
    assert "val_interval = 10" in full_text
    assert "initial_num_query = 200" in full_text
    assert "grow_num_query = 100" in full_text
    assert "grow_every_epochs = 25" in full_text
    assert 'raw_depth_from_images = _bool_env("ADAOCC_RAW_DEPTH_FROM_IMAGES", False)' in full_text
    # Same model architecture as the released mini baseline.
    for token in (
        "RADIOHFBackbone",
        "SparseEncoderTPVOnly",
        "TPVLiteEncoder",
        "AdaOccTransformer",
        "AdaOccHead",
    ):
        assert token in full_text

    readme = (ROOT / "README.md").read_text()
    for doc in [
        "README.md",
        "docs/REPRODUCIBILITY.md",
        "docs/AI_REPRODUCTION.md",
        "docs/DATA.md",
        "docs/ARCHITECTURE.md",
        "docs/DEPENDENCY_TRACE.md",
    ]:
        assert "configs/occscannet/radio_occscannet_full.py" in (ROOT / doc).read_text(), doc
    assert "train_occscannet_full.pkl" in readme
    assert "checkpoints/adaocc_radio_occscannet_full_epoch100.pth" in readme
    # Full-split labels/depth must be generated explicitly, not only for the mini PKLs.
    full_splits = "--splits train_occscannet_full.pkl val_occscannet_full.pkl test_occscannet_full.pkl"
    assert full_splits in readme
    assert full_splits in (ROOT / "docs" / "DATA.md").read_text()
    assert full_splits in (ROOT / "docs" / "AI_REPRODUCTION.md").read_text()
    assert full_splits in (ROOT / "docs" / "REPRODUCIBILITY.md").read_text()

    result = run_cmd([PYTHON, str(checker), "--help"])
    assert result.returncode == 0, result.stderr + result.stdout
    assert "--config" in result.stdout
    assert "--checkpoint" in result.stdout
