"""Render saved targets using original CPU or depth-ordered CUDA bilinear splats."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.target_cameras import scene_identity, validate_camera
from pipeline.point_cloud_renderer import PointCloudRenderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--cameras', required=True, type=Path, help='target_cameras.json')
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--frame_ids', nargs='+', type=int, help='Default: render all saved targets')
    parser.add_argument('--render_device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--render_batch_size', type=int, default=1000000)
    parser.add_argument('--render_max_splats', type=int, default=32000000)
    args = parser.parse_args()
    if args.render_batch_size <= 0 or args.render_max_splats <= 0:
        parser.error('render_batch_size and render_max_splats must be positive')
    if args.render_device == 'cuda':
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable in this process; check nvidia-smi and PyTorch')
    document = json.loads(args.cameras.read_text())
    cameras = [validate_camera(c) for c in document['cameras']]
    selected = args.frame_ids if args.frame_ids is not None else list(range(len(cameras)))
    if not selected or any(i < 0 or i >= len(cameras) for i in selected) or len(set(selected)) != len(selected):
        parser.error('Choose distinct frame IDs within the saved target range')
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error('output_dir must be empty or nonexistent')
    with np.load(args.scene, allow_pickle=False) as scene:
        identity = scene_identity(scene['input_c2w'], scene['intrinsics_reconstruction'], scene['image_size_reconstruction'])
        if identity != document['scene_id']:
            parser.error('Targets belong to a different Pi3 world coordinate frame')
        print('Loading FULL point cloud into CPU RAM (not the browser preview subset).', flush=True)
        points, colors = scene['points'], scene['colors']
    renderer = PointCloudRenderer(points, colors, args.render_device, args.render_batch_size, args.render_max_splats)
    for folder in ['rgb', 'mask']:
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    for frame_id in selected:
        camera = cameras[frame_id]
        print(f'Rendering target {frame_id} with {len(points):,} points ({args.render_device})', flush=True)
        rgb, mask = renderer.render(camera)
        for folder, image in [('rgb', rgb), ('mask', mask)]:
            if not cv2.imwrite(str(args.output_dir / folder / f'target_{frame_id:06d}.png'), image):
                raise OSError('Failed to write render output')
    (args.output_dir / 'cameras.json').write_text(json.dumps(
        dict(scene_id=identity, render_device=args.render_device,
             cameras=[dict(frame_id=i, **cameras[i]) for i in selected]), indent=2) + '\n')
    renderer.close()
    print(f'Saved target point-cloud RGB conditions and masks to {args.output_dir}')


if __name__ == '__main__':
    main()
