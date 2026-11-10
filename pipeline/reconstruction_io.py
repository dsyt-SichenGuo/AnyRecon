"""Export the exact rendering cloud and its camera coordinate system."""
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def write_colored_ply(path, points, colors):
    vertices = np.empty(len(points), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    for axis, name in enumerate(['x', 'y', 'z']):
        vertices[name] = points[:, axis]
    for channel, name in enumerate(['red', 'green', 'blue']):
        vertices[name] = colors[:, channel]
    PlyData([PlyElement.describe(vertices, 'vertex')], byte_order='<').write(str(path))


def center_crop_transform(image_size, output_size):
    """Pixel transform matching run_pi3.center_crop_to_size (width, height)."""
    width, height = map(int, image_size)
    out_width, out_height = map(int, output_size)
    resized_width, resized_height = width, height
    if width < out_width or height < out_height:
        scale = max(out_width / width, out_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
    left = max(0, (resized_width - out_width) // 2)
    top = max(0, (resized_height - out_height) // 2)
    return np.array([
        [resized_width / width, 0, -left],
        [0, resized_height / height, -top],
        [0, 0, 1],
    ], dtype=np.float32)


def save_reconstruction(output_dir, *, points, colors, source_indices,
                        input_c2w, render_w2c, intrinsics, reference_rgb,
                        frame_sources, original_sizes, num_reference_frames,
                        image_size, output_size, intrinsics_estimated,
                        settings, preview_points=200000):
    """Keep world coordinates unchanged; retain only reference-derived points."""
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    source_indices = np.asarray(source_indices, dtype=np.int32)
    input_c2w = np.asarray(input_c2w, dtype=np.float32)
    render_w2c = np.asarray(render_w2c, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    reference_rgb = np.asarray(reference_rgb, dtype=np.uint8)
    count = len(frame_sources)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('Cannot export an empty or malformed reference point cloud')
    if colors.shape != points.shape or source_indices.shape != (len(points),):
        raise ValueError('Point colors and source indices must match the rendering cloud')
    if not np.isfinite(points).all():
        raise ValueError('Reference point cloud contains non-finite coordinates')
    if not 0 < num_reference_frames <= count:
        raise ValueError('Invalid reference count')
    if source_indices.min() < 0 or source_indices.max() >= num_reference_frames:
        raise ValueError('Point sources must refer only to reference frames')
    if input_c2w.shape != (count, 4, 4) or render_w2c.shape != (count, 4, 4):
        raise ValueError('Expected one 4x4 pose per input frame')
    if intrinsics.shape != (count, 3, 3):
        raise ValueError('Expected one 3x3 intrinsics matrix per input frame')
    if reference_rgb.shape != (num_reference_frames, image_size[1], image_size[0], 3):
        raise ValueError('Reference RGB images must use the reconstruction image size')
    if np.asarray(original_sizes).shape != (count, 2):
        raise ValueError('Expected original width/height for every frame')
    if preview_points <= 0:
        raise ValueError('preview_points must be positive')

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    input_w2c = np.linalg.inv(input_c2w)
    render_c2w = np.linalg.inv(render_w2c)
    pixel_transform = center_crop_transform(image_size, output_size)
    output_intrinsics = pixel_transform[None] @ intrinsics
    np.savez_compressed(
        output / 'scene.npz', schema_version=np.int32(1),
        points=points, colors=colors, source_indices=source_indices,
        input_c2w=input_c2w, input_w2c=input_w2c,
        render_c2w=render_c2w, render_w2c=render_w2c,
        intrinsics_reconstruction=intrinsics, intrinsics_output=output_intrinsics,
        reconstruction_to_output_pixels=pixel_transform,
        image_size_reconstruction=np.asarray(image_size, dtype=np.int32),
        image_size_output=np.asarray(output_size, dtype=np.int32),
        original_image_sizes=np.asarray(original_sizes, dtype=np.int32),
        reference_rgb=reference_rgb,
        num_reference_frames=np.int32(num_reference_frames),
        frame_sources=np.asarray(frame_sources, dtype=str),
    )
    write_colored_ply(output / 'reference_points.ply', points, colors)
    preview_ids = np.linspace(0, len(points) - 1, min(len(points), preview_points), dtype=np.int64)
    write_colored_ply(output / 'preview_points.ply', points[preview_ids], colors[preview_ids])
    camera_colors = np.tile(np.array([255, 160, 0], dtype=np.uint8), (count, 1))
    camera_colors[:num_reference_frames] = [0, 220, 80]
    write_colored_ply(output / 'camera_centers.ply', render_c2w[:, :3, 3], camera_colors)

    cameras = []
    for index, source in enumerate(frame_sources):
        cameras.append(dict(
            frame_id=index, source=source, is_reference=index < num_reference_frames,
            c2w_estimated=input_c2w[index].tolist(), w2c_estimated=input_w2c[index].tolist(),
            c2w_render=render_c2w[index].tolist(), w2c_render=render_w2c[index].tolist(),
            K_reconstruction=intrinsics[index].tolist(), K_output=output_intrinsics[index].tolist(),
        ))
    (output / 'cameras.json').write_text(json.dumps(cameras, indent=2) + '\n')
    metadata = dict(
        schema_version=1, num_points=len(points), preview_num_points=len(preview_ids),
        num_reference_frames=num_reference_frames, num_target_frames=count - num_reference_frames,
        world_coordinates='Unmodified Pi3 world frame from this inference; no recentering or rescaling.',
        units='Arbitrary reconstruction scale; not calibrated meters.',
        camera_coordinates='OpenCV: +X right, +Y down, +Z forward.',
        renderer_extrinsics='w2c: X_camera = R @ X_world + t; c2w is its inverse.',
        point_selection='Only reference-frame points passing the same confidence/depth-edge masks used for rendering.',
        image_size_reconstruction=list(map(int, image_size)),
        image_size_output=list(map(int, output_size)),
        intrinsics_estimated=bool(intrinsics_estimated),
        output_intrinsics_note='K_output = reconstruction_to_output_pixels @ K_reconstruction; uses the pipeline resize/crop calibration convention.',
        render_pose_note='Reference poses are unchanged; target render poses may be trajectory-smoothed. Estimated poses are also retained.',
        camera_center_colors=dict(reference='green', target='orange'),
        bounds_min=points.min(axis=0).tolist(), bounds_max=points.max(axis=0).tolist(),
        settings=settings,
    )
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Saved reference cloud ({len(points)} points), cameras and reusable cache to {output}')
    return output
