"""
Epipolar Consistency Check for NeRF camera poses.

Verifies camera poses by checking epipolar geometry constraints:
for each pair of neighboring cameras, feature matches should lie
on their corresponding epipolar lines if the poses are correct.

Usage:
    from verification import verify_dataset
    results = verify_dataset(scene_path, cameras, K, dataset_type)
"""

import cv2
import numpy as np

MAX_IMG_DIM = 800
SIFT_FEATURES = 1000
RATIO_THRESH = 0.7
NUM_NEIGHBORS = 3

# Thresholds (pixels) for grading
THRESH_GOOD = 2.0
THRESH_WARNING = 5.0


def get_intrinsics_synthetic(scene_path, camera_angle_x, img_w=800, img_h=800):
    """Build K matrix for synthetic datasets from camera_angle_x."""
    focal = 0.5 * img_w / np.tan(0.5 * camera_angle_x)
    cx, cy = img_w / 2.0, img_h / 2.0
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)


def get_intrinsics_llff(h, w, focal):
    """Build K matrix for LLFF/real360 datasets from pose metadata."""
    scale = min(1.0, MAX_IMG_DIM / max(h, w))
    focal_s = focal * scale
    cx = (w * scale) / 2.0
    cy = (h * scale) / 2.0
    return np.array([[focal_s, 0, cx], [0, focal_s, cy], [0, 0, 1]], dtype=np.float64)


def _skew(t):
    """Skew-symmetric matrix of a 3-vector."""
    return np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0],
    ], dtype=np.float64)


def _synthetic_c2w_to_opencv(mat_4x4):
    """Convert synthetic NeRF c2w (OpenGL: Y-up, -Z-forward) to OpenCV (Y-down, Z-forward).

    Negates Y and Z columns of the rotation.
    """
    M = np.array(mat_4x4, dtype=np.float64)
    M[:3, 1] *= -1
    M[:3, 2] *= -1
    return M


def _llff_c2w_to_opencv(mat_4x4):
    """Convert LLFF c2w to OpenCV convention.

    LLFF stores rotation columns as [down, right, back].
    OpenCV c2w columns should be [right, down, forward].
    """
    M = np.array(mat_4x4, dtype=np.float64)
    R = M[:3, :3].copy()
    t = M[:3, 3].copy()
    out = np.eye(4)
    out[:3, 0] = R[:, 1]    # right
    out[:3, 1] = R[:, 0]    # down
    out[:3, 2] = -R[:, 2]   # forward = -back
    out[:3, 3] = t
    return out


def compute_fundamental(mat_i, mat_j, K, dataset_type="synthetic"):
    """Compute Fundamental matrix from two 4x4 c2w matrices and K.

    Handles coordinate convention conversion based on dataset_type.
    """
    if dataset_type == "llff":
        M_i = _llff_c2w_to_opencv(mat_i)
        M_j = _llff_c2w_to_opencv(mat_j)
    else:
        M_i = _synthetic_c2w_to_opencv(mat_i)
        M_j = _synthetic_c2w_to_opencv(mat_j)

    R_i, t_i = M_i[:3, :3], M_i[:3, 3]
    R_j, t_j = M_j[:3, :3], M_j[:3, 3]

    R_rel = R_j.T @ R_i
    t_rel = R_j.T @ (t_i - t_j)

    E = _skew(t_rel) @ R_rel
    K_inv = np.linalg.inv(K)
    return K_inv.T @ E @ K_inv


def _match_sift(img_i, img_j):
    """SIFT detection + ratio test + mutual consistency check.

    Returns (pts1, pts2) as Nx2 arrays, or (None, None) if insufficient matches.
    """
    sift = cv2.SIFT_create(nfeatures=SIFT_FEATURES)
    kp1, des1 = sift.detectAndCompute(img_i, None)
    kp2, des2 = sift.detectAndCompute(img_j, None)

    if des1 is None or des2 is None or len(des1) < 5 or len(des2) < 5:
        return None, None

    bf = cv2.BFMatcher(cv2.NORM_L2)

    # Forward ratio test
    raw12 = bf.knnMatch(des1, des2, k=2)
    good12 = {}
    for pair in raw12:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_THRESH * n.distance:
                good12[m.queryIdx] = m.trainIdx

    # Backward ratio test
    raw21 = bf.knnMatch(des2, des1, k=2)
    good21 = {}
    for pair in raw21:
        if len(pair) == 2:
            m, n = pair
            if m.distance < RATIO_THRESH * n.distance:
                good21[m.queryIdx] = m.trainIdx

    # Mutual consistency
    pts1, pts2 = [], []
    for i1, i2 in good12.items():
        if i2 in good21 and good21[i2] == i1:
            pts1.append(kp1[i1].pt)
            pts2.append(kp2[i2].pt)

    if len(pts1) < 5:
        return None, None

    return np.float32(pts1), np.float32(pts2)


def compute_epipolar_error(img_path_i, img_path_j, F):
    """Match features and compute median symmetric epipolar error.

    Uses SIFT with ratio test and mutual consistency for robust matching.
    Returns (median_error, num_matches) or (None, 0) if insufficient matches.
    """
    img_i = cv2.imread(str(img_path_i), cv2.IMREAD_GRAYSCALE)
    img_j = cv2.imread(str(img_path_j), cv2.IMREAD_GRAYSCALE)
    if img_i is None or img_j is None:
        return None, 0

    def resize(img):
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMG_DIM:
            s = MAX_IMG_DIM / max(h, w)
            return cv2.resize(img, (int(w * s), int(h * s)))
        return img

    img_i = resize(img_i)
    img_j = resize(img_j)

    pts1, pts2 = _match_sift(img_i, img_j)
    if pts1 is None:
        return None, 0

    # Symmetric epipolar distance
    errors = []
    for i in range(len(pts1)):
        p1 = np.array([pts1[i, 0], pts1[i, 1], 1.0])
        p2 = np.array([pts2[i, 0], pts2[i, 1], 1.0])

        Fp1 = F @ p1
        d1 = abs(p2 @ Fp1) / max(np.sqrt(Fp1[0] ** 2 + Fp1[1] ** 2), 1e-8)

        Ftp2 = F.T @ p2
        d2 = abs(p1 @ Ftp2) / max(np.sqrt(Ftp2[0] ** 2 + Ftp2[1] ** 2), 1e-8)

        errors.append((d1 + d2) / 2.0)

    return float(np.median(errors)), len(errors)


MAX_CAMERAS = 100  # sample limit for large datasets


def verify_dataset(scene_path, cameras, K, dataset_type="synthetic"):
    """Run epipolar verification on a list of cameras.

    Args:
        scene_path: Path to dataset root (for resolving image paths).
        cameras: List of camera dicts with 'position', 'matrix', 'image' keys.
        K: 3x3 intrinsic matrix (numpy array).
        dataset_type: 'synthetic' or 'llff' (affects coordinate convention).

    Returns:
        List of dicts: [{ index, error, grade, pairs: [...] }, ...]
        For large datasets, only a sampled subset is verified; unverified
        cameras get grade="unverified".
    """
    n = len(cameras)
    if n < 2:
        return []

    # Sample if dataset is too large
    if n > MAX_CAMERAS:
        rng = np.random.RandomState(42)
        sampled_indices = set(rng.choice(n, MAX_CAMERAS, replace=False).tolist())
    else:
        sampled_indices = set(range(n))

    positions = np.array([c["position"] for c in cameras])

    # Build unique pairs: each sampled camera with its K nearest neighbors
    pair_errors = {}
    for i in sorted(sampled_indices):
        dists = np.linalg.norm(positions - positions[i], axis=1)
        dists[i] = np.inf
        # Pick nearest neighbors that are also in the sampled set
        sorted_nn = np.argsort(dists)
        neighbors = [int(j) for j in sorted_nn if j in sampled_indices][:NUM_NEIGHBORS]

        for j in neighbors:
            key = (min(i, j), max(i, j))
            if key in pair_errors:
                continue

            F = compute_fundamental(
                cameras[i]["matrix"], cameras[j]["matrix"], K, dataset_type
            )
            img_i = scene_path / cameras[i]["image"]
            img_j = scene_path / cameras[j]["image"]

            err, num_matches = compute_epipolar_error(img_i, img_j, F)
            pair_errors[key] = {"error": err, "matches": num_matches}

    # Aggregate per-camera
    results = []
    for i in range(n):
        if i not in sampled_indices:
            results.append({
                "index": i,
                "error": -1,
                "grade": "unverified",
                "pairs": [],
            })
            continue

        cam_pairs = []
        cam_errors = []
        for (a, b), info in pair_errors.items():
            if a == i or b == i:
                other = b if a == i else a
                if info["error"] is not None:
                    cam_errors.append(info["error"])
                    cam_pairs.append({
                        "neighbor": other,
                        "error": round(info["error"], 4),
                        "matches": info["matches"],
                    })

        median_error = float(np.median(cam_errors)) if cam_errors else -1

        if median_error < 0:
            grade = "unknown"
        elif median_error < THRESH_GOOD:
            grade = "good"
        elif median_error < THRESH_WARNING:
            grade = "warning"
        else:
            grade = "bad"

        results.append({
            "index": i,
            "error": round(median_error, 4),
            "grade": grade,
            "pairs": cam_pairs,
        })

    return results
