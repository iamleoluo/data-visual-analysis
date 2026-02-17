# NeRF Camera Pose Viewer & Verifier

Interactive 3D viewer for NeRF datasets with automated **epipolar geometry verification** to detect incorrect camera poses.

![3D Viewer](https://img.shields.io/badge/3D-Three.js-blue) ![Backend](https://img.shields.io/badge/Backend-Flask-green) ![CV](https://img.shields.io/badge/CV-OpenCV-red)

## Features

- **3D Visualization** - View camera positions and orientations in an interactive Three.js scene
- **Image Preview** - Click any camera to preview its image and pose matrix
- **Split Filtering** - Filter cameras by train/val/test splits
- **Epipolar Verification** - Automated pose quality check using epipolar geometry constraints
- **Color-coded Results** - Cameras colored green/yellow/red based on verification grade

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place datasets

Put NeRF datasets under `dataset/`:

```
dataset/
├── nerf_example_data/nerf_synthetic/   # Synthetic datasets (lego, chair, etc.)
│   └── lego/
│       ├── transforms_train.json
│       ├── transforms_val.json
│       ├── transforms_test.json
│       └── train/, val/, test/         # Image folders
├── nerf_llff_data/                     # LLFF datasets (fern, flower, etc.)
│   └── fern/
│       ├── poses_bounds.npy
│       └── images/
└── nerf_real_360/                      # Real 360 datasets
    └── pinecone/
        ├── poses_bounds.npy
        └── images/
```

### 3. Run

```bash
python app.py
```

Open http://localhost:8080

## Usage

### Viewing

1. Select a dataset from the dropdown
2. **Scroll** to zoom, **left-drag** to rotate, **right-drag** to pan
3. **Click** a camera point (sphere) to see its image and pose info
4. Use checkboxes to filter train/val/test splits

### Verification

1. Load a dataset
2. Click **Verify Poses** (takes ~10-60s depending on dataset size)
3. Cameras change color based on their epipolar error:
   - **Green** (good): < 2 px - pose is accurate
   - **Yellow** (warning): 2-5 px - minor inaccuracy detected
   - **Red** (bad): > 5 px - significant pose error
   - **Dark gray** (skipped): not sampled (for datasets > 100 cameras)
4. Click any camera to see detailed error info and neighbor pair data
5. Click **Reset Colors** to return to split-based coloring

## How Verification Works

The verification module (`verification/epipolar.py`) checks camera pose consistency using epipolar geometry:

1. For each camera, find its **3 nearest neighbors** (by position)
2. For each pair, compute the **Fundamental matrix** from known poses and intrinsics
3. Detect **SIFT features** in both images, match with ratio test + mutual consistency
4. Measure the **symmetric epipolar distance** for each match (how far each point is from its expected epipolar line)
5. The **median error** across matches is the pair's score; each camera's final score is the median across its pairs

If the poses are correct, matched points should lie precisely on their epipolar lines (error near 0). Large errors indicate the camera pose may be wrong.

### Coordinate Conventions

The module handles two different coordinate systems:
- **Synthetic NeRF**: OpenGL convention (Y-up, -Z-forward)
- **LLFF/Real360**: Columns stored as `[down, right, back]`

Both are converted to OpenCV convention internally before computing the geometry.

## Project Structure

```
.
├── app.py                    # Flask server, API routes, dataset parsing
├── static/index.html         # Single-page frontend (Three.js + vanilla JS)
├── verification/             # Standalone verification module
│   ├── __init__.py
│   └── epipolar.py           # Epipolar consistency check algorithm
├── requirements.txt
└── dataset/                  # NeRF datasets (not tracked in git)
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/datasets` | List all detected datasets |
| `GET /api/dataset/<path>` | Get camera poses for a dataset |
| `GET /api/image/<path>` | Serve a dataset image |
| `GET /api/verify/<path>` | Run epipolar verification, returns per-camera results |

## Supported Datasets

| Format | Detection | Example |
|--------|-----------|---------|
| NeRF Synthetic | `transforms_train.json` present | lego, chair, drums, hotdog |
| LLFF | `poses_bounds.npy` present | fern, flower, fortress, horns |
| Real 360 | `poses_bounds.npy` present | pinecone, vasedeck |

## Requirements

- Python 3.8+
- Flask
- NumPy
- OpenCV (`opencv-python`)
