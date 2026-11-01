"""Copy-move forgery localisation using SIFT keypoint self-matching.

A copy-move forgery copies one region of an image and pastes it elsewhere in
the same image. Keypoints inside the copied region will have near-identical
descriptors to keypoints in the pasted region, but sit at a different
location. We match the image's keypoints against themselves (excluding the
trivial self-match), keep matches that pass Lowe's ratio test and are far
enough apart, then keep only groups of matches that share a consistent
displacement, which removes most random false matches.
"""
import cv2
import numpy as np


def detect_copy_move(image_bgr, ratio=0.6, min_distance=60,
                     min_cluster=8, max_side=1024):
    """Return dict with matched point pairs and an annotated image."""
    img = image_bgr.copy()
    scale = 1.0
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=4000)
    keypoints, desc = sift.detectAndCompute(gray, None)

    result = {"pairs": [], "suspicious": False, "annotated": img,
              "num_keypoints": len(keypoints)}
    if desc is None or len(keypoints) < 10:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    # k=3: the first neighbour is the keypoint itself
    knn = matcher.knnMatch(desc, desc, k=3)

    candidates = []
    for m in knn:
        if len(m) < 3:
            continue
        _, best, second = m
        if best.distance < ratio * second.distance:
            p1 = np.array(keypoints[best.queryIdx].pt)
            p2 = np.array(keypoints[best.trainIdx].pt)
            if np.linalg.norm(p1 - p2) > min_distance:
                candidates.append((p1, p2))

    # Each pair is found twice (a->b and b->a); normalise direction
    norm = []
    seen = set()
    for p1, p2 in candidates:
        a, b = (p1, p2) if tuple(p1) < tuple(p2) else (p2, p1)
        key = (round(a[0]), round(a[1]), round(b[0]), round(b[1]))
        if key not in seen:
            seen.add(key)
            norm.append((a, b))

    # Group by displacement vector: true copy-move pairs share one shift
    groups = {}
    for a, b in norm:
        d = b - a
        key = (int(round(d[0] / 8)), int(round(d[1] / 8)))
        groups.setdefault(key, []).append((a, b))

    pairs = [p for g in groups.values() if len(g) >= min_cluster for p in g]

    annotated = img.copy()
    for a, b in pairs:
        pa = tuple(int(v) for v in a)
        pb = tuple(int(v) for v in b)
        cv2.line(annotated, pa, pb, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(annotated, pa, 4, (0, 0, 255), -1)
        cv2.circle(annotated, pb, 4, (255, 0, 0), -1)

    result.update({
        "pairs": [(a / scale, b / scale) for a, b in pairs],
        "suspicious": len(pairs) >= min_cluster,
        "annotated": annotated,
    })
    return result
