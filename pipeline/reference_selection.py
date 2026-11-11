"""Occlusion-aware sampled-cloud coverage for selecting chunk reference RGBs."""
import numpy as np


def visible_source_coverage(points, source_indices, camera, num_references,
                            width=320, depth_tolerance=0.01, candidate_ids=None):
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
    candidates = np.arange(num_references) if candidate_ids is None else np.asarray(candidate_ids, dtype=np.int64)
    mapping = np.full(num_references, -1, dtype=np.int64)
    mapping[candidates] = np.arange(len(candidates))
    selected_sources = mapping[sources[visible]]
    eligible = selected_sources >= 0
    coverage = np.zeros((len(candidates), width * height), dtype=bool)
    coverage[selected_sources[eligible], flat[visible][eligible]] = True
    return coverage


def original_source_counts(points, colors, source_indices, camera, num_references, block_size=1000000):
    """Match run_pi3's pre-blending in-frustum, nonblack source counts on ALL points.

    The upstream count includes occluded points; it is not a visibility z-buffer.
    Blocks bound temporary projection memory without subsampling the point cloud.
    """
    E, K = np.asarray(camera['w2c']), np.asarray(camera['K'])
    width, height = camera['image_size']
    counts = np.zeros(num_references, dtype=np.int64)
    for start in range(0, len(points), block_size):
        end = min(len(points), start + block_size)
        xyz = np.asarray(points[start:end], dtype=np.float64) @ E[:3, :3].T + E[:3, 3]
        keep = (xyz[:, 2] > 0.1) & (colors[start:end].sum(axis=1) > 0)
        projected = xyz[keep] @ K.T
        uv = projected[:, :2] / (projected[:, 2:3] + 1e-8)
        inside = ((uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5)
                  & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5))
        counts += np.bincount(source_indices[start:end][keep][inside], minlength=num_references)
    return counts


def choose_original_references(counts, count, top_per_target=10):
    """Upstream: per-target top-10, sum contributions, chunk top-K, then ID order."""
    aggregate = {}
    for row in np.asarray(counts):
        ranked = sorted(np.flatnonzero(row).tolist(), key=lambda i: -int(row[i]))[:top_per_target]
        for source in ranked:
            aggregate[source] = aggregate.get(source, 0) + int(row[source])
    selected = sorted(sorted(aggregate, key=lambda i: -aggregate[i])[:count])
    if not selected:
        raise ValueError('No candidate reference contributes projected points to this chunk')
    return selected, dict(aggregate_source_counts={str(i): value for i, value in aggregate.items()},
                         top_per_target=top_per_target)


def coverage_diagnostic(masks, selected):
    """One shared sampled-visibility metric for comparing both selection methods."""
    available = masks.any(axis=1).sum(axis=1)
    covered = masks[:, selected].any(axis=1).sum(axis=1)
    fractions = covered / np.maximum(1, available)
    return dict(selected_coverage_fraction_per_target=fractions.tolist(),
                sampled_visible_pixels_per_target=available.tolist(),
                mean_coverage=float(fractions.mean()), min_coverage=float(fractions.min()),
                targets_without_sampled_candidate_coverage=int(np.sum(available == 0)))


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
