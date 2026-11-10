"""Reconstruct a reusable Pi3 scene using reference images only."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as TF

from geometry_utils import depth_edge, recover_focal_shift, intrinsics_from_focal_center
from Pi3.pi3.models.pi3 import Pi3
from pipeline.reconstruction_io import save_reconstruction


def select_reference_paths(directory, num_frames=None):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f'Reference directory does not exist: {directory}')
    paths = sorted(p for p in directory.iterdir()
                   if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not paths:
        raise ValueError(f'No reference images found directly in {directory}')
    if num_frames is not None:
        if not 1 <= num_frames <= len(paths):
            raise ValueError(f'num_frames must be in 1..{len(paths)}')
        indices = ([0] if num_frames == 1 else
                   [round(i * (len(paths) - 1) / (num_frames - 1)) for i in range(num_frames)])
        paths = [paths[i] for i in indices]
    return paths


def target_resolution(image_size, pixel_limit=255000, patch=14):
    """Same resize-budget calculation as the original run_pi3.py."""
    width, height = image_size
    if min(width, height) <= 0:
        raise ValueError('Image dimensions must be positive')
    if pixel_limit < patch * patch:
        raise ValueError(f'pixel_limit must be at least {patch * patch}')
    scale = math.sqrt(pixel_limit / (width * height))
    target_width, target_height = width * scale, height * scale
    k, m = round(target_width / patch), round(target_height / patch)
    while k * patch * m * patch > pixel_limit:
        if k / m > target_width / target_height:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch, max(1, m) * patch


def load_reference_images(paths, pixel_limit):
    """Match the original BGR -> RGB -> PIL LANCZOS -> ToTensor path."""
    images, original_sizes = [], []
    to_tensor = TF.ToTensor()
    size = None
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode reference image: {path}')
        original_sizes.append((image.shape[1], image.shape[0]))
        pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if size is None:
            size = target_resolution(pil.size, pixel_limit)
        images.append(to_tensor(pil.resize(size, Image.Resampling.LANCZOS)))
    return torch.stack(images), original_sizes, size


def reconstruct_references(model, images, device, confidence_threshold=0.1, depth_edge_rtol=0.03):
    """Original inference/filter/color logic with every input treated as reference."""
    device = torch.device(device)
    images = images.to(device)
    if device.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        context = torch.amp.autocast('cuda', dtype=dtype)
    else:
        context = nullcontext()
    with torch.no_grad(), context:
        result = model(images[None])

    confidence_mask = torch.sigmoid(result['conf'][..., 0]) > confidence_threshold
    masks_batched = confidence_mask & ~depth_edge(result['local_points'][..., 2], rtol=depth_edge_rtol)
    masks = masks_batched[0]
    points = result['points'][0][masks].float().cpu().numpy()
    # Multiplication and uint8 conversion match the original script.
    colors = (images.permute(0, 2, 3, 1)[masks].cpu().numpy() * 255).astype(np.uint8)
    source_map = torch.arange(len(images), device=device).view(-1, 1, 1).expand(masks.shape)
    source_indices = source_map[masks].cpu().numpy()
    if not len(points):
        raise ValueError('No reference points passed the confidence/depth-edge filters')

    height, width = images.shape[-2:]
    intrinsics_estimated, intrinsics_error = True, None
    try:
        local_points = result['local_points']
        focal, _ = recover_focal_shift(local_points, masks_batched, downsample_size=(64, 64))
        aspect_ratio = width / height
        fx = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio * width
        fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 * height
        intrinsics = intrinsics_from_focal_center(fx, fy, width / 2, height / 2)[0].float().cpu().numpy()
    except Exception as error:
        # Preserve the original fallback, but clearly identify it in the cache.
        intrinsics_estimated, intrinsics_error = False, str(error)
        print(f'Intrinsics prediction failed; using original fallback: {error}', flush=True)
        fallback = np.array([[width, 0, width / 2], [0, width, height / 2], [0, 0, 1]], dtype=np.float32)
        intrinsics = np.repeat(fallback[None], len(images), axis=0)
    c2w = result['camera_poses'][0].float().cpu().numpy()
    reference_rgb = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    return dict(points=points, colors=colors, source_indices=source_indices,
                input_c2w=c2w, render_w2c=np.linalg.inv(c2w), intrinsics=intrinsics,
                reference_rgb=reference_rgb, intrinsics_estimated=intrinsics_estimated,
                intrinsics_error=intrinsics_error)


def reconstruct_and_export(model, paths, output_dir, *, device='cuda', pixel_limit=255000,
                           confidence_threshold=0.1, depth_edge_rtol=0.03,
                           preview_points=200000, settings=None, viewer=True):
    images, original_sizes, image_size = load_reference_images(paths, pixel_limit)
    print(f'Reconstructing {len(paths)} reference images at {image_size[0]}x{image_size[1]}', flush=True)
    result = reconstruct_references(model, images, device, confidence_threshold, depth_edge_rtol)
    extra = dict(settings or {}, pixel_limit=pixel_limit,
                 confidence_threshold=confidence_threshold, depth_edge_rtol=depth_edge_rtol,
                 intrinsics_error=result.pop('intrinsics_error'), renderer_near_clip=0.1,
                 mode='reference_only', reference_rgb='Exact resized Pi3 RGB inputs, uint8.')
    output = save_reconstruction(
        output_dir, **result, frame_sources=[str(Path(p).resolve()) for p in paths],
        original_sizes=original_sizes, num_reference_frames=len(paths),
        image_size=image_size, output_size=image_size, settings=extra,
        preview_points=preview_points,
    )
    references = [dict(reference_id=i, image=str(Path(path).resolve()),
                       original_size=list(original_sizes[i])) for i, path in enumerate(paths)]
    (output / 'references.json').write_text(json.dumps(references, indent=2) + '\n')
    if viewer:
        from pipeline.reconstruction_visualization import write_scene_viewer
        write_scene_viewer(output / 'scene.npz', output / 'viewer.html', max_points=preview_points)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_dir', required=True, type=Path,
                        help='Directory containing only reference PNG/JPG images; no target frames')
    parser.add_argument('--num_frames', type=int,
                        help='Uniformly select N images from this directory; omit to use all images')
    parser.add_argument('--ckpt', type=Path, default=Path('Pi3/model.safetensors'))
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--pixel_limit', type=int, default=255000)
    parser.add_argument('--confidence_threshold', type=float, default=0.1)
    parser.add_argument('--depth_edge_rtol', type=float, default=0.03)
    parser.add_argument('--preview_points', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_viewer', action='store_true', help='Export PLY/cache without the HTML viewer')
    args = parser.parse_args()
    if args.pixel_limit < 196 or args.preview_points <= 0:
        parser.error('pixel_limit must be >=196 and preview_points must be positive')
    if not 0 < args.confidence_threshold < 1 or args.depth_edge_rtol <= 0:
        parser.error('confidence_threshold must be in (0,1); depth_edge_rtol must be positive')
    try:
        paths = select_reference_paths(args.reference_dir, args.num_frames)
    except ValueError as error:
        parser.error(str(error))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output_dir is not empty; select a new directory to preserve existing results')
    if not args.ckpt.is_file():
        parser.error(f'Model checkpoint not found: {args.ckpt}')
    if not args.no_viewer:
        import plotly  # Check the viewer dependency before expensive GPU reconstruction.
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable in this process')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f'Loading Pi3 from {args.ckpt}', flush=True)
    model = Pi3().to(args.device).eval()
    if args.ckpt.suffix == '.safetensors':
        from safetensors.torch import load_file
        weights = load_file(str(args.ckpt))
    else:
        weights = torch.load(str(args.ckpt), map_location='cpu', weights_only=False)
    model.load_state_dict(weights)
    del weights
    output = reconstruct_and_export(
        model, paths, args.output_dir, device=args.device, pixel_limit=args.pixel_limit,
        confidence_threshold=args.confidence_threshold, depth_edge_rtol=args.depth_edge_rtol,
        preview_points=args.preview_points,
        settings=dict(checkpoint=str(args.ckpt.resolve()), seed=args.seed),
        viewer=not args.no_viewer,
    )
    print(f'Done. Full point cloud: {output / "reference_points.ply"}', flush=True)
    if not args.no_viewer:
        print(f'Open offline interactive viewer: {output / "viewer.html"}', flush=True)


if __name__ == '__main__':
    main()
