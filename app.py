import json
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_file

from verification import verify_dataset
from verification.epipolar import get_intrinsics_synthetic, get_intrinsics_llff

app = Flask(__name__, static_folder="static")

DATASET_ROOT = Path(__file__).parent / "dataset"


def parse_synthetic(scene_path):
    """Parse NeRF synthetic dataset (transforms JSON format)."""
    cameras = []
    for split in ("train", "val", "test"):
        tf_path = scene_path / f"transforms_{split}.json"
        if not tf_path.exists():
            continue
        with open(tf_path) as f:
            data = json.load(f)
        for frame in data.get("frames", []):
            mat = frame["transform_matrix"]
            # Position is the translation column (column 3, rows 0-2)
            pos = [mat[0][3], mat[1][3], mat[2][3]]
            # Camera forward direction (negative z-axis of the camera)
            forward = [-mat[0][2], -mat[1][2], -mat[2][2]]
            # Build image path relative to scene directory
            file_path = frame["file_path"]
            # file_path is like ./train/r_0 — append .png
            rel = file_path.lstrip("./")
            if not rel.endswith(".png"):
                rel += ".png"
            cameras.append({
                "position": pos,
                "forward": forward,
                "matrix": mat,
                "image": rel,
                "split": split,
            })
    return cameras


def parse_llff(scene_path):
    """Parse LLFF / real-360 dataset (poses_bounds.npy format)."""
    pb_path = scene_path / "poses_bounds.npy"
    if not pb_path.exists():
        return []

    poses_bounds = np.load(str(pb_path))  # (N, 17)
    n = poses_bounds.shape[0]

    # Get sorted image list
    images_dir = scene_path / "images"
    if not images_dir.exists():
        return []
    image_files = sorted(
        f.name for f in images_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    cameras = []
    for i in range(min(n, len(image_files))):
        row = poses_bounds[i]
        # First 15 values reshape to 3x5 (LLFF format: [R | t | hwf])
        pose = row[:15].reshape(3, 5)
        # In LLFF convention the columns are:
        # col 0-2: rotation (right, up, back)  col 3: translation  col 4: [h, w, f]
        # Position (translation vector)
        pos = pose[:, 3].tolist()
        # Forward direction: negative of the third column (back -> forward)
        forward = (-pose[:, 2]).tolist()
        # Build a 4x4 matrix for display
        mat = np.eye(4)
        mat[:3, :4] = pose[:, :4]
        cameras.append({
            "position": pos,
            "forward": forward,
            "matrix": mat.tolist(),
            "image": f"images/{image_files[i]}",
            "split": "all",
        })
    return cameras


def scan_datasets():
    """Scan dataset/ directory for all available NeRF datasets."""
    datasets = []
    if not DATASET_ROOT.exists():
        return datasets

    # Synthetic datasets
    synthetic_root = DATASET_ROOT / "nerf_example_data" / "nerf_synthetic"
    if synthetic_root.exists():
        for scene in sorted(synthetic_root.iterdir()):
            if scene.is_dir() and (scene / "transforms_train.json").exists():
                rel = scene.relative_to(DATASET_ROOT)
                datasets.append({
                    "name": f"synthetic / {scene.name}",
                    "path": str(rel),
                    "type": "synthetic",
                })

    # LLFF datasets
    llff_root = DATASET_ROOT / "nerf_llff_data"
    if llff_root.exists():
        for scene in sorted(llff_root.iterdir()):
            if scene.is_dir() and (scene / "poses_bounds.npy").exists():
                rel = scene.relative_to(DATASET_ROOT)
                datasets.append({
                    "name": f"llff / {scene.name}",
                    "path": str(rel),
                    "type": "llff",
                })

    # Real 360 datasets
    real360_root = DATASET_ROOT / "nerf_real_360"
    if real360_root.exists():
        for scene in sorted(real360_root.iterdir()):
            if scene.is_dir() and (scene / "poses_bounds.npy").exists():
                rel = scene.relative_to(DATASET_ROOT)
                datasets.append({
                    "name": f"real360 / {scene.name}",
                    "path": str(rel),
                    "type": "real360",
                })

    return datasets


def _detect_type(scene_path):
    """Detect dataset type: 'synthetic' or 'llff'."""
    if (scene_path / "transforms_train.json").exists():
        return "synthetic"
    elif (scene_path / "poses_bounds.npy").exists():
        return "llff"
    return None


def _build_intrinsics(scene_path, dataset_type):
    """Build intrinsic matrix K based on dataset type."""
    if dataset_type == "synthetic":
        with open(scene_path / "transforms_train.json") as f:
            data = json.load(f)
        return get_intrinsics_synthetic(scene_path, data["camera_angle_x"])
    else:
        poses_bounds = np.load(str(scene_path / "poses_bounds.npy"))
        pose = poses_bounds[0, :15].reshape(3, 5)
        h, w, focal = pose[:, 4]
        return get_intrinsics_llff(h, w, focal)


def _parse_cameras(scene_path, dataset_type):
    """Parse cameras based on dataset type."""
    if dataset_type == "synthetic":
        return parse_synthetic(scene_path)
    else:
        return parse_llff(scene_path)


# ── Routes ──

@app.route("/")
def index():
    return send_file("static/index.html")


@app.route("/api/datasets")
def list_datasets():
    return jsonify(scan_datasets())


@app.route("/api/dataset/<path:dataset_path>")
def get_dataset(dataset_path):
    scene_path = DATASET_ROOT / dataset_path
    if not scene_path.exists():
        return jsonify({"error": "Dataset not found"}), 404

    dataset_type = _detect_type(scene_path)
    if dataset_type is None:
        return jsonify({"error": "Unknown dataset format"}), 400

    cameras = _parse_cameras(scene_path, dataset_type)
    return jsonify({"cameras": cameras, "count": len(cameras)})


@app.route("/api/image/<path:image_path>")
def serve_image(image_path):
    full_path = DATASET_ROOT / image_path
    if not full_path.exists():
        return jsonify({"error": "Image not found"}), 404
    return send_file(str(full_path))


@app.route("/api/verify/<path:dataset_path>")
def verify(dataset_path):
    scene_path = DATASET_ROOT / dataset_path
    if not scene_path.exists():
        return jsonify({"error": "Dataset not found"}), 404

    dataset_type = _detect_type(scene_path)
    if dataset_type is None:
        return jsonify({"error": "Unknown dataset format"}), 400

    cameras = _parse_cameras(scene_path, dataset_type)
    K = _build_intrinsics(scene_path, dataset_type)

    try:
        results = verify_dataset(scene_path, cameras, K, dataset_type)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(debug=True, port=8080)
