"""Render saved free-camera targets with the original Pi3 splat renderer (CPU)."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.target_cameras import scene_identity, validate_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--cameras', required=True, type=Path, help='target_cameras.json')
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--frame_ids', nargs='+', type=int, help='Default: render all saved targets')
    args = parser.parse_args()
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
    from run_pi3 import render_point_cloud_to_memory
    for folder in ['rgb', 'mask']:
        (args.output_dir / folder).mkdir(parents=True, exist_ok=True)
    for frame_id in selected:
        camera = cameras[frame_id]
        print(f'Rendering target {frame_id} with {len(points):,} points', flush=True)
        rgb, masks, _ = render_point_cloud_to_memory(
            points, colors, np.asarray([camera['w2c']]), np.asarray([camera['K']]),
            tuple(camera['image_size']))
        for folder, image in [('rgb', rgb[0]), ('mask', masks[0])]:
            if not cv2.imwrite(str(args.output_dir / folder / f'target_{frame_id:06d}.png'), image):
                raise OSError('Failed to write render output')
    (args.output_dir / 'cameras.json').write_text(json.dumps(
        dict(scene_id=identity, cameras=[dict(frame_id=i, **cameras[i]) for i in selected]), indent=2) + '\n')
    print(f'Saved target point-cloud RGB conditions and masks to {args.output_dir}')


if __name__ == '__main__':
    main()
