"""Build AnyRecon chunks from a cached cloud and hand-picked cameras."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.reference_selection import visible_source_coverage, choose_chunk_references
from pipeline.target_cameras import scene_identity, validate_camera


def dump_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def read_reference(path, size):
    rgb = cv2.imread(str(path))
    if rgb is None:
        raise ValueError(f'Cannot read reference RGB: {path}')
    return cv2.resize(rgb, tuple(size), interpolation=cv2.INTER_LANCZOS4)


def contact_sheet(paths, labels, output):
    sheet = Image.new('RGB', (320 * 3, 268 * ((len(paths) + 2) // 3)), '#18232d')
    draw = ImageDraw.Draw(sheet)
    for i, (path, label) in enumerate(zip(paths, labels)):
        with Image.open(path) as source:
            image = source.convert('RGB')
            image.thumbnail((320, 240))
        x, y = i % 3 * 320, i // 3 * 268
        sheet.paste(image, (x, y))
        draw.text((x + 5, y + 243), label, fill='white')
    sheet.save(output)


def verified_video(path, frames, save_video):
    """Atomic video output; partial encodes are not mistaken for completed chunks."""
    temporary = path.with_name(path.stem + '.tmp.mp4')
    if not save_video(frames, str(temporary), fps=10):
        raise RuntimeError(f'Video encoding failed: {path}')
    cap = cv2.VideoCapture(str(temporary))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if count != len(frames) or (width, height) != (frames[0].shape[1], frames[0].shape[0]):
        raise RuntimeError(f'Invalid encoded condition video: {temporary}')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--cameras', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--chunk_size', type=int, default=4)
    parser.add_argument('--num_reference', type=int, default=6)
    parser.add_argument('--retrieval_points', type=int, default=2000000)
    parser.add_argument('--retrieval_width', type=int, default=320)
    parser.add_argument('--depth_tolerance', type=float, default=0.01)
    parser.add_argument('--context_chunks', action='store_true', help='Plan for two-frame context carry between sequential chunks')
    parser.add_argument('--plan_only', action='store_true', help='Select references and write contact sheets; skip full rendering')
    parser.add_argument('--chunk_ids', nargs='+', type=int, help='Prepare only these chunks; default all')
    args = parser.parse_args()
    if min(args.chunk_size, args.num_reference, args.retrieval_points, args.retrieval_width) < 1 or not np.isfinite(args.depth_tolerance) or args.depth_tolerance < 0:
        parser.error('Counts must be positive and depth_tolerance finite/nonnegative')
    document = json.loads(args.cameras.read_text())
    cameras = [validate_camera(c) for c in document['cameras']]
    if not cameras or len({tuple(c['image_size']) for c in cameras}) != 1:
        parser.error('Targets must be nonempty and share one output size')
    size = cameras[0]['image_size']
    settings = dict(scene_path=str(args.scene.resolve()),
                    scene_size=args.scene.stat().st_size, scene_mtime_ns=args.scene.stat().st_mtime_ns,
                    cameras_sha256=hashlib.sha256(args.cameras.read_bytes()).hexdigest(),
                    chunk_size=args.chunk_size, num_reference=args.num_reference,
                    retrieval_points=args.retrieval_points, retrieval_width=args.retrieval_width,
                    depth_tolerance=args.depth_tolerance)
    if args.context_chunks:
        settings['context_chunks'] = True
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / 'manifest.json'
    with np.load(args.scene, allow_pickle=False) as scene:
        identity = scene_identity(scene['input_c2w'], scene['intrinsics_reconstruction'], scene['image_size_reconstruction'])
        if identity != document['scene_id']:
            parser.error('Target cameras belong to a different scene coordinate system')
        reference_paths = scene['frame_sources'].tolist()[:int(scene['num_reference_frames'])]
        if args.num_reference > len(reference_paths):
            parser.error('num_reference must fit source reference pool')
        for path in reference_paths:
            if not Path(path).is_file():
                parser.error(f'Missing original reference image: {path}')
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest['settings'] != settings:
                parser.error('Existing output has different settings/cameras; use a new output_dir')
            print('Reusing existing reference selection manifest.', flush=True)
        else:
            if any(root.iterdir()):
                parser.error('Output directory is nonempty without a manifest; choose a new directory')
            print('Loading sampled cloud for occlusion-aware reference retrieval.', flush=True)
            points = scene['points']
            ids = np.linspace(0, len(points) - 1, min(len(points), args.retrieval_points), dtype=np.int64)
            sample = points[ids].astype(np.float32)
            del points
            sources = scene['source_indices'][ids]
            colors = scene['colors'][ids]
            keep = colors.sum(axis=1) > 0
            sample, sources = sample[keep], sources[keep]
            del colors
            manifest = dict(schema_version=1, scene_id=identity, settings=settings,
                            image_size=size, independent_chunks=not args.context_chunks, target_cameras=cameras,
                            retrieval_note='Uniform point sample; coarse z-buffer; per-target best seed then normalized pixel coverage gain. Approximate visibility, not full-cloud visibility.',
                            chunks=[])
            for start in range(0, len(cameras), args.chunk_size):
                target_ids = list(range(start, min(start + args.chunk_size, len(cameras))))
                masks = np.stack([visible_source_coverage(sample, sources, cameras[i], len(reference_paths),
                                  args.retrieval_width, args.depth_tolerance) for i in target_ids])
                selection, diagnostic = choose_chunk_references(masks, args.num_reference)
                del masks
                # Keep strongest selected reference first for CLIP image conditioning.
                reference_ids = sorted(selection, key=lambda i: (-sum(diagnostic['selected_pixels_per_target'][t][selection.index(i)] for t in range(len(target_ids))), i))
                chunk = dict(chunk_id=len(manifest['chunks']), target_ids=target_ids,
                             reference_ids=reference_ids, reference_paths=[reference_paths[i] for i in reference_ids],
                             video=f'selected_chunk_{len(manifest["chunks"]):03d}.mp4', retrieval=diagnostic)
                manifest['chunks'].append(chunk)
                print(f'Chunk {chunk["chunk_id"]}: targets={target_ids}, references={reference_ids}', flush=True)
            del sample, sources
            dump_json(manifest_path, manifest)
    for folder in ['condition', 'mask', 'reference_sheets', 'renders/rgb', 'renders/mask', 'status']:
        (root / folder).mkdir(parents=True, exist_ok=True)
    dump_json(root / 'target_cameras.json', document)
    selected_chunks = args.chunk_ids if args.chunk_ids is not None else list(range(len(manifest['chunks'])))
    if len(set(selected_chunks)) != len(selected_chunks) or any(i < 0 or i >= len(manifest['chunks']) for i in selected_chunks):
        parser.error('Invalid or duplicate chunk_ids')
    for chunk in manifest['chunks']:
        contact_sheet(chunk['reference_paths'],
                      [f'ref {i}: {Path(p).name}' for i, p in zip(chunk['reference_ids'], chunk['reference_paths'])],
                      root / 'reference_sheets' / f'chunk_{chunk["chunk_id"]:03d}.jpg')
    if args.plan_only:
        print(f'Reference selection ready: {manifest_path}; inspect reference_sheets/', flush=True)
        return
    todo = [manifest['chunks'][i] for i in selected_chunks if not (root / 'status' / f'chunk_{i:03d}.json').exists()]
    if not todo:
        print('Requested chunks already prepared.', flush=True)
        return
    from run_pi3 import render_point_cloud_to_memory, save_video
    print('Loading FULL cloud for original splat renderer; CPU RAM usage can be tens of GB.', flush=True)
    with np.load(args.scene, allow_pickle=False) as scene:
        points, colors = scene['points'], scene['colors']
    for chunk in todo:
        rgb_frames = [read_reference(path, size) for path in chunk['reference_paths']]
        mask_frames = [np.full((size[1], size[0], 3), 255, dtype=np.uint8) for _ in rgb_frames]
        coverage = []
        for target_id in chunk['target_ids']:
            camera = cameras[target_id]
            rgb_path = root / 'renders/rgb' / f'target_{target_id:06d}.png'
            mask_path = root / 'renders/mask' / f'target_{target_id:06d}.png'
            rgb, mask = cv2.imread(str(rgb_path)) if rgb_path.exists() else None, cv2.imread(str(mask_path)) if mask_path.exists() else None
            if rgb is None or mask is None:
                print(f'Rendering target {target_id} from all {len(points):,} points.', flush=True)
                frames, masks, _ = render_point_cloud_to_memory(points, colors, np.asarray([camera['w2c']]),
                                                               np.asarray([camera['K']]), tuple(size))
                rgb, mask = frames[0], masks[0]
                if not cv2.imwrite(str(rgb_path), rgb) or not cv2.imwrite(str(mask_path), mask):
                    raise OSError('Failed to save target render')
            if rgb.shape != (size[1], size[0], 3) or mask.shape != rgb.shape:
                raise ValueError('Cached render size mismatch')
            fraction = float(np.mean(mask[:, :, 0] > 0))
            if fraction == 0:
                raise ValueError(f'Target {target_id} has an empty full-cloud mask')
            coverage.append(fraction)
            rgb_frames.append(rgb)
            mask_frames.append(mask)
        verified_video(root / 'condition' / chunk['video'], rgb_frames, save_video)
        verified_video(root / 'mask' / chunk['video'], mask_frames, save_video)
        (root / 'condition' / (Path(chunk['video']).stem + '_info.txt')).write_text(
            f'Condition Frame Count: {len(chunk["reference_ids"])}\n'
            f'Render Frame Count: {len(chunk["target_ids"])}\n'
            f'Condition Frame Indices: {chunk["reference_ids"]}\n')
        dump_json(root / 'status' / f'chunk_{chunk["chunk_id"]:03d}.json',
                  dict(target_ids=chunk['target_ids'], mask_coverage=coverage))
        print(f'Prepared {chunk["video"]}; target mask coverage={coverage}', flush=True)
    print(f'AnyRecon input ready: {root}', flush=True)


if __name__ == '__main__':
    main()
