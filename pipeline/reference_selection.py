"""Occlusion-aware sampled-cloud coverage for selecting chunk reference RGBs."""
import numpy as np


def visible_source_coverage(points, source_indices, camera, num_references,
                            width=320, depth_tolerance=0.01):
    """Return source x pixel masks using a coarse nearest-depth buffer.

    This is a retrieval approximation, never the cloud used for final rendering.
    Nearby samples within depth_tolerance of the front surface may all contribute.
    """
    out_width, out_height = camera['image_size']
    height = max(1, round(width * out_height / out_width))
    K = np.asarray(camera['K'], dtype=np.float32).copy()
    K[0] *= width / out_width
    K[1] *= height / out_height
    E = np.asarray(camera['w2c'], dtype=np.float32)
    xyz = points @ E[:3, :3].T + E[:3, 3]
    front = xyz[:, 2] > 0.1
    xyz, sources = xyz[front], source_indices[front]
    projected = xyz @ K.T
    uv = projected[:, :2] / projected[:, 2:3]
    pixels = np.floor(uv + 0.5).astype(np.int64)
    inside = (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    pixels, sources, depth = pixels[inside], sources[inside], xyz[inside, 2]
    flat = pixels[:, 1] * width + pixels[:, 0]
    nearest = np.full(width * height, np.inf, dtype=np.float32)
    np.minimum.at(nearest, flat, depth)
    visible = depth <= nearest[flat] * (1 + depth_tolerance) + 1e-6
    coverage = np.zeros((num_references, width * height), dtype=bool)
    coverage[sources[visible], flat[visible]] = True
    return coverage


def choose_chunk_references(coverage, count):
    """Seed each target's strongest reference, then add normalized coverage gain."""
    masks = np.asarray(coverage, dtype=bool)  # target, reference, pixel
    targets, references, pixels = masks.shape
    if not 1 <= count <= references:
        raise ValueError('Reference count must fit reference pool')
    counts = masks.sum(axis=2)
    if np.any(counts.max(axis=1) == 0):
        raise ValueError('A target has no sampled visible points; inspect pose/coverage before generation')
    selected = list(dict.fromkeys(np.argmax(counts, axis=1).tolist())) if count >= targets else []
    covered = masks[:, selected].any(axis=1) if selected else np.zeros((targets, pixels), dtype=bool)
    normalization = np.maximum(1, masks.any(axis=1).sum(axis=1))
    while len(selected) < count:
        # Avoid allocating another full target x source x pixel tensor.
        score = np.zeros(references, dtype=np.float64)
        for target in range(targets):
            gain = (masks[target] & ~covered[target]).sum(axis=1) / normalization[target]
            # When there are fewer reference slots than targets, favor targets
            # whose sampled visible surfaces have received less coverage so far.
            weight = 1.0 if count >= targets else 1 / (0.05 + covered[target].sum() / normalization[target])
            score += gain * weight
        # Deterministic secondary preference if no additional pixels can be covered.
        total = (counts / normalization[:, None]).sum(axis=0)
        available = [i for i in range(references) if i not in selected]
        best = max(available, key=lambda i: (score[i], total[i], -i))
        selected.append(best)
        covered |= masks[:, best]
    return selected, dict(
        selection_order=selected,
        best_reference_per_target=np.argmax(counts, axis=1).tolist(),
        sampled_visible_pixels_per_target=masks.any(axis=1).sum(axis=1).tolist(),
        selected_coverage_fraction_per_target=(covered.sum(axis=1) / normalization).tolist(),
        selected_pixels_per_target=counts[:, selected].tolist(),
    )
