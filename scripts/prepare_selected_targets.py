"""Render recorded targets and package AnyRecon inputs with selectable reference retrieval."""
import argparse
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import shlex
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.reference_selection import (
    visible_source_coverage, choose_chunk_references, original_source_counts,
    choose_original_references, coverage_diagnostic,
)
from pipeline.target_cameras import scene_identity, validate_camera, target_chunk_groups


def dump_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
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
    temporary = path.with_name(path.stem + '.tmp.mp4')
    if not save_video(frames, str(temporary), fps=10):
        raise RuntimeError(f'Video encoding failed: {path}')
    if not video_matches(temporary, len(frames), [frames[0].shape[1], frames[0].shape[0]]):
        raise RuntimeError(f'Invalid encoded condition video: {temporary}')
    temporary.replace(path)


def video_matches(path, count, size):
    if not path.is_file():
        return False
    cap = cv2.VideoCapture(str(path))
    valid = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == count
             and [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))] == size)
    cap.release()
    return valid


def resolve_reference_pool(reference_paths, *, reference_ids=None, reference_pool=None,
                           reference_pool_size=None, reference_dir=None):
    """Pool IDs always refer to the scene's original reference IDs, never renumbered IDs."""
    overrides = {}
    if reference_pool is not None:
        if reference_pool.is_dir():
            if reference_dir is not None:
                raise ValueError('A reference_pool directory supplies its own RGB paths; omit reference_dir')
            by_name = {}
            for index, path in enumerate(reference_paths):
                by_name.setdefault(Path(path).name, []).append(index)
            images = sorted(p for p in reference_pool.iterdir()
                            if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
            reference_ids = []
            for image in images:
                matches = by_name.get(image.name, [])
                if len(matches) != 1:
                    raise ValueError(f'Candidate image must match exactly one reconstructed reference: {image.name}')
                reference_ids.append(matches[0])
                overrides[matches[0]] = image
        elif reference_pool.is_file():
            if reference_pool.suffix.lower() == '.json':
                value = json.loads(reference_pool.read_text())
                reference_ids = value['reference_ids'] if isinstance(value, dict) else value
            else:
                tokens = ' '.join(line.split('#', 1)[0] for line in reference_pool.read_text().splitlines())
                reference_ids = [int(token) for token in tokens.replace(',', ' ').split()]
        else:
            raise ValueError(f'Reference pool does not exist: {reference_pool}')
    if reference_ids is None:
        reference_ids = list(range(len(reference_paths)))
    if (not isinstance(reference_ids, list) or not reference_ids
            or any(type(i) is not int or not 0 <= i < len(reference_paths) for i in reference_ids)
            or len(set(reference_ids)) != len(reference_ids)):
        raise ValueError('Candidate reference IDs must be unique integers within the scene reference pool')
    ids = sorted(reference_ids)
    if reference_pool_size is not None:
        if not 1 <= reference_pool_size <= len(ids):
            raise ValueError('reference_pool_size must fit the selected candidate pool')
        ids = [ids[i] for i in np.rint(np.linspace(0, len(ids) - 1, reference_pool_size)).astype(int)]
    if reference_dir is not None and not reference_dir.is_dir():
        raise ValueError(f'Reference RGB directory does not exist: {reference_dir}')
    pool = []
    for index in ids:
        source = reference_paths[index]
        path = overrides.get(index, reference_dir / Path(source).name if reference_dir else Path(source)).resolve()
        if not path.is_file():
            raise ValueError(f'Missing candidate reference {index}: {path}; use --reference_dir to relocate original RGBs')
        stat = path.stat()
        pool.append(dict(reference_id=index, source_path=source, path=str(path),
                         size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns))
    resolved = [entry['path'] for entry in pool]
    if len(set(resolved)) != len(resolved):
        raise ValueError('Multiple reference IDs resolve to the same RGB file; preserve unambiguous source paths')
    return pool


def select_references(method, masks, counts, candidate_ids, count, top_per_target):
    if method == 'original':
        selected, diagnostic = choose_original_references(counts, count, top_per_target)
        diagnostic['aggregate_source_counts'] = {
            str(candidate_ids[int(i)]): value for i, value in diagnostic['aggregate_source_counts'].items()}
    else:
        selected, diagnostic = choose_chunk_references(masks, count)
        # Strongest selected image is first for image/CLIP conditioning, as in our improved workflow.
        contribution = masks.sum(axis=(0, 2))
        selected = sorted(selected, key=lambda i: (-int(contribution[i]), candidate_ids[i]))
        for field in ['selection_order', 'best_reference_per_target']:
            diagnostic[field] = [candidate_ids[i] for i in diagnostic[field]]
    diagnostic.update(coverage_diagnostic(masks, selected))
    diagnostic['candidate_reference_ids'] = candidate_ids
    return [candidate_ids[i] for i in selected], diagnostic


def chunk_description(chunk_id, target_ids, reference_ids, paths, method, diagnostic):
    frames = [dict(reference_id=index, source_path=paths[index],
                   image=f'references/chunk_{chunk_id:03d}/rgb/ref_{i:03d}_source_{index:06d}.png',
                   mask=f'references/chunk_{chunk_id:03d}/mask/ref_{i:03d}_source_{index:06d}.png')
              for i, index in enumerate(reference_ids)]
    return dict(chunk_id=chunk_id, target_ids=target_ids, reference_ids=reference_ids,
                reference_paths=[paths[i] for i in reference_ids], reference_frames=frames,
                camera_metadata=f'cameras/chunk_{chunk_id:03d}.json',
                video=f'selected_chunk_{chunk_id:03d}.mp4', reference_method=method, retrieval=diagnostic)


def write_comparison(root, manifests, method_roots):
    report = dict(metric_note='Same sampled full-scene z-buffer and candidate pool for both methods. '
                  'Coverage is a geometric proxy, not measured generation quality.', chunks=[])
    rows = []
    for i in range(len(next(iter(manifests.values()))['chunks'])):
        choices = {}
        for method, manifest in manifests.items():
            chunk = manifest['chunks'][i]
            metric = chunk['retrieval']
            choices[method] = dict(reference_ids=chunk['reference_ids'], actual_reference_count=len(chunk['reference_ids']),
                                   mean_coverage=metric['mean_coverage'], min_coverage=metric['min_coverage'],
                                   per_target_coverage=metric['selected_coverage_fraction_per_target'])
            sheet = str((method_roots[method] / 'reference_sheets' / f'chunk_{i:03d}.jpg').relative_to(root))
            rows.append(f'<tr><td>{i}</td><td>{escape(method)}</td><td>{escape(str(chunk["reference_ids"]))}</td>'
                        f'<td>{metric["mean_coverage"]:.3%}</td><td>{metric["min_coverage"]:.3%}</td>'
                        f'<td><a href="{escape(sheet, quote=True)}"><img width="400" src="{escape(sheet, quote=True)}"></a></td></tr>')
        preferred = max(choices, key=lambda m: (choices[m]['min_coverage'], choices[m]['mean_coverage']))
        item = dict(chunk_id=i, methods=choices, coverage_proxy_preference=preferred)
        if len(choices) == 2:
            a, b = (set(c['reference_ids']) for c in choices.values())
            item['reference_overlap_jaccard'] = len(a & b) / max(1, len(a | b))
            scores = {(c['min_coverage'], c['mean_coverage']) for c in choices.values()}
            if len(scores) == 1:
                item['coverage_proxy_preference'] = 'tie'
        report['chunks'].append(item)
    dump_json(root / 'selection_comparison.json', report)
    (root / 'selection_comparison.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>参考帧选择对比</title>'
        '<style>body{font:15px system-ui;margin:24px;background:#f5f7fa;color:#18232d}'
        'table{border-collapse:collapse}td,th{border:1px solid #ccd4dd;padding:10px;vertical-align:top}img{max-width:100%}</style>'
        '<h1>参考帧选择对比</h1><p>original：原项目的全点云投影贡献计数；improved：近似遮挡后的覆盖增益。'
        '两者用相同采样可见性指标评估。覆盖率是几何代理指标，不能直接证明最终生成质量更好。</p>'
        '<table><tr><th>Chunk</th><th>方法</th><th>参考 ID</th><th>平均覆盖</th><th>最差目标覆盖</th><th>参考图</th></tr>'
        + ''.join(rows) + '</table></html>')


def render_backend_matches(render_root, target_id, device):
    metadata = render_root / 'renders/metadata' / f'target_{target_id:06d}.json'
    expected = 'cpu_original_v1' if device == 'cpu' else 'cuda_alpha_over_v1'
    if not metadata.exists():
        return device == 'cpu'  # Legacy files were always made by the CPU renderer.
    try:
        return json.loads(metadata.read_text()).get('backend') == expected
    except (ValueError, OSError):
        return False


def chunk_complete(root, chunk, size, render_root=None, render_device='cpu'):
    required = [root / 'status' / f'chunk_{chunk["chunk_id"]:03d}.json',
                root / chunk['camera_metadata'], root / 'condition' / (Path(chunk['video']).stem + '_info.txt')]
    required += [root / frame[field] for frame in chunk['reference_frames'] for field in ['image', 'mask']]
    render_root = root if render_root is None else render_root
    required += [render_root / 'renders' / folder / f'target_{i:06d}.png'
                 for i in chunk['target_ids'] for folder in ['rgb', 'mask']]
    count = len(chunk['reference_ids']) + len(chunk['target_ids'])
    status = root / 'status' / f'chunk_{chunk["chunk_id"]:03d}.json'
    try:
        if json.loads(status.read_text()).get('render_device', 'cpu') != render_device:
            return False
    except (ValueError, OSError):
        return False
    return (all(path.is_file() for path in required)
            and all(render_backend_matches(render_root, i, render_device) for i in chunk['target_ids'])
            and all(video_matches(root / folder / chunk['video'], count, size) for folder in ['condition', 'mask']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path, help='scene.npz: cloud, source IDs, reference cameras and RGB paths')
    parser.add_argument('--cameras', required=True, type=Path, help='Recorded target_cameras.json')
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--chunk_size', type=int, default=4, help='Legacy files only; recorded chunk boundaries take precedence')
    parser.add_argument('--num_reference', type=int, default=6, help='Improved: exact count; original: upper limit, matching upstream')
    parser.add_argument('--reference_method', choices=['original', 'improved', 'compare'], default='improved')
    pool_group = parser.add_mutually_exclusive_group()
    pool_group.add_argument('--reference_ids', nargs='+', type=int, help='Candidate reference IDs (zero based, from references.json)')
    pool_group.add_argument('--reference_pool', type=Path, help='JSON/TXT ID list, or a directory of reconstructed reference RGBs')
    parser.add_argument('--reference_pool_size', type=int, help='Uniformly reduce the candidate pool to this many images')
    parser.add_argument('--reference_dir', type=Path, help='Relocated original RGB directory; match the scene filenames')
    parser.add_argument('--original_top_per_target', type=int, default=10, help='Upstream per-target shortlist length (original default: 10)')
    parser.add_argument('--retrieval_points', type=int, default=2000000, help='Improved retrieval and shared comparison metric only')
    parser.add_argument('--retrieval_width', type=int, default=320)
    parser.add_argument('--depth_tolerance', type=float, default=0.01)
    parser.add_argument('--render_device', choices=['cpu', 'cuda'], default='cpu', help='Target rendering only; reference retrieval is unchanged')
    parser.add_argument('--render_batch_size', type=int, default=1000000, help='CUDA projection batch size; does not cap visible-splat sort memory')
    parser.add_argument('--context_chunks', action='store_true')
    parser.add_argument('--plan_only', action='store_true', help='Select references and write comparison sheets; skip full rendering/packaging')
    parser.add_argument('--chunk_ids', nargs='+', type=int, help='Render/package only these chunks; default all')
    args = parser.parse_args()
    if (min(args.chunk_size, args.num_reference, args.original_top_per_target, args.retrieval_points, args.retrieval_width, args.render_batch_size) < 1
            or not np.isfinite(args.depth_tolerance) or args.depth_tolerance < 0):
        parser.error('Counts must be positive and depth_tolerance finite/nonnegative')
    if args.render_device == 'cuda' and not args.plan_only:
        import torch
        if not torch.cuda.is_available():
            parser.error('CUDA is unavailable in this process; check nvidia-smi and PyTorch, or use --render_device cpu')
    document = json.loads(args.cameras.read_text())
    cameras = [validate_camera(c) for c in document['cameras']]
    if not cameras or len({tuple(c['image_size']) for c in cameras}) != 1:
        parser.error('Targets must be nonempty and share one output size')
    try:
        target_groups = target_chunk_groups(document, args.chunk_size)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    selected_chunks = args.chunk_ids if args.chunk_ids is not None else list(range(len(target_groups)))
    if len(set(selected_chunks)) != len(selected_chunks) or any(i < 0 or i >= len(target_groups) for i in selected_chunks):
        parser.error('Invalid or duplicate chunk_ids')
    size = cameras[0]['image_size']
    with np.load(args.scene, allow_pickle=False) as scene:
        input_c2w = scene['input_c2w']
        input_K = scene['intrinsics_reconstruction']
        input_size = scene['image_size_reconstruction'].tolist()
        identity = scene_identity(input_c2w, input_K, input_size)
        if identity != document['scene_id']:
            parser.error('Target cameras belong to a different scene coordinate system')
        reference_paths = scene['frame_sources'].tolist()[:int(scene['num_reference_frames'])]
    try:
        pool = resolve_reference_pool(reference_paths, reference_ids=args.reference_ids, reference_pool=args.reference_pool,
                                      reference_pool_size=args.reference_pool_size, reference_dir=args.reference_dir)
    except (ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    if args.num_reference > len(pool):
        parser.error('num_reference must fit the selected candidate reference pool')
    candidate_ids = [entry['reference_id'] for entry in pool]
    resolved_paths = {entry['reference_id']: entry['path'] for entry in pool}
    methods = ['original', 'improved'] if args.reference_method == 'compare' else [args.reference_method]
    settings = dict(preparation_version=2, scene_path=str(args.scene.resolve()),
                    scene_size=args.scene.stat().st_size, scene_mtime_ns=args.scene.stat().st_mtime_ns,
                    cameras_sha256=hashlib.sha256(args.cameras.read_bytes()).hexdigest(),
                    chunk_size=args.chunk_size, num_reference=args.num_reference,
                    retrieval_points=args.retrieval_points, retrieval_width=args.retrieval_width,
                    depth_tolerance=args.depth_tolerance, original_top_per_target=args.original_top_per_target,
                    reference_pool=pool, methods=methods)
    if args.context_chunks:
        settings['context_chunks'] = True
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'preparation_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != settings:
            parser.error('Existing output has different settings/cameras/RGBs; use a new output_dir')
    elif any(root.iterdir()):
        parser.error('Output is nonempty without preparation_config.json; choose a new directory')
    else:
        dump_json(config_path, settings)
    dump_json(root / 'reference_pool.json', dict(candidates=pool, count=len(pool), reference_id_note='Original scene IDs'))
    method_roots = {method: root / method if len(methods) > 1 else root for method in methods}
    manifests = {}
    missing = []
    for method, output in method_roots.items():
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'manifest.json'
        if path.exists():
            manifest = json.loads(path.read_text())
            if manifest['settings'] != settings:
                parser.error('Existing manifest has different settings; choose a new output_dir')
            manifests[method] = manifest
        else:
            missing.append(method)
            manifests[method] = dict(schema_version=2, scene_id=identity, settings=settings, image_size=size,
                                     independent_chunks=not args.context_chunks, target_cameras=cameras,
                                     reference_method=method, chunks=[])
    full = {}

    def full_cloud(with_sources=False):
        with np.load(args.scene, allow_pickle=False) as scene:
            for name in ['points', 'colors'] + (['source_indices'] if with_sources else []):
                if name not in full:
                    full[name] = scene[name]
        return full

    if missing:
        print(f'Candidate pool: {len(pool)} / {len(reference_paths)}; methods={methods}', flush=True)
        if 'original' in missing:
            print('Original retrieval counts ALL cloud points (including occluded points), as in run_pi3.py.', flush=True)
            full_cloud(with_sources=True)
        with np.load(args.scene, allow_pickle=False) as scene:
            points = full['points'] if 'points' in full else scene['points']
            sample_ids = np.linspace(0, len(points) - 1, min(len(points), args.retrieval_points), dtype=np.int64)
            sample = points[sample_ids].astype(np.float32)
            del points
            sources = full['source_indices'][sample_ids] if 'source_indices' in full else scene['source_indices'][sample_ids]
            colors = full['colors'][sample_ids] if 'colors' in full else scene['colors'][sample_ids]
            keep = colors.sum(axis=1) > 0
            sample, sources = sample[keep], sources[keep]
            del colors
        for chunk_id, target_ids in enumerate(target_groups):
            masks = np.stack([visible_source_coverage(sample, sources, cameras[i], len(reference_paths),
                              args.retrieval_width, args.depth_tolerance, candidate_ids=candidate_ids) for i in target_ids])
            counts = None
            if 'original' in missing:
                counts = np.stack([original_source_counts(full['points'], full['colors'], full['source_indices'],
                                   cameras[i], len(reference_paths))[candidate_ids] for i in target_ids])
            for method in missing:
                ids, diagnostic = select_references(method, masks, counts, candidate_ids,
                                                    args.num_reference, args.original_top_per_target)
                chunk = chunk_description(chunk_id, target_ids, ids, resolved_paths, method, diagnostic)
                manifests[method]['chunks'].append(chunk)
                print(f'{method}: chunk {chunk_id}, {len(target_ids)} targets, refs={ids}, '
                      f'mean/min coverage={diagnostic["mean_coverage"]:.3f}/{diagnostic["min_coverage"]:.3f}', flush=True)
            del masks, counts
        del sample, sources
        for method in missing:
            dump_json(method_roots[method] / 'manifest.json', manifests[method])
    for method, output in method_roots.items():
        for folder in ['condition', 'mask', 'reference_sheets', 'status', 'cameras']:
            (output / folder).mkdir(parents=True, exist_ok=True)
        dump_json(output / 'target_cameras.json', document)
        for chunk in manifests[method]['chunks']:
            contact_sheet(chunk['reference_paths'], [f'ref {i}: {Path(p).name}' for i, p in
                          zip(chunk['reference_ids'], chunk['reference_paths'])],
                          output / 'reference_sheets' / f'chunk_{chunk["chunk_id"]:03d}.jpg')
    write_comparison(root, manifests, method_roots)
    if args.plan_only:
        print(f'Reference plans ready: {root / "selection_comparison.html"}; no condition videos rendered.', flush=True)
        return
    todo = {method: [manifests[method]['chunks'][i] for i in selected_chunks
                     if not chunk_complete(output, manifests[method]['chunks'][i], size, root, args.render_device)]
            for method, output in method_roots.items()}
    if not any(todo.values()):
        print('Requested chunks already prepared.', flush=True)
        return
    from run_pi3 import save_video
    from pipeline.point_cloud_renderer import PointCloudRenderer
    renderer = None
    rgb_dir, mask_dir = root / 'renders/rgb', root / 'renders/mask'
    rgb_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    for target_id in sorted({i for chunks in todo.values() for chunk in chunks for i in chunk['target_ids']}):
        rgb_path, mask_path = rgb_dir / f'target_{target_id:06d}.png', mask_dir / f'target_{target_id:06d}.png'
        rgb = cv2.imread(str(rgb_path)) if rgb_path.exists() else None
        mask = cv2.imread(str(mask_path)) if mask_path.exists() else None
        if (rgb is None or mask is None or rgb.shape != (size[1], size[0], 3) or mask.shape != rgb.shape
                or not render_backend_matches(root, target_id, args.render_device)):
            cloud = full_cloud()
            if renderer is None:
                renderer = PointCloudRenderer(cloud['points'], cloud['colors'], args.render_device, args.render_batch_size)
            print(f'Rendering target {target_id} using ALL {len(cloud["points"]):,} points ({args.render_device}).', flush=True)
            camera = cameras[target_id]
            started = time.perf_counter()
            rgb, mask = renderer.render(camera)
            if not cv2.imwrite(str(rgb_path), rgb) or not cv2.imwrite(str(mask_path), mask):
                raise OSError('Failed to save target render')
            dump_json(root / 'renders/metadata' / f'target_{target_id:06d}.json',
                      dict(backend=renderer.backend, render_seconds=time.perf_counter()-started,
                           point_count=len(cloud['points']), image_size=size))
            print(f'Target {target_id} rendered in {time.perf_counter()-started:.2f}s', flush=True)
        if not np.any(mask[:, :, 0] > 0):
            raise ValueError(f'Target {target_id} has an empty full-cloud mask; inspect pose before generation')
    if renderer is not None:
        renderer.close()
    full.clear()
    white = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    for method, chunks in todo.items():
        output = method_roots[method]
        for chunk in chunks:
            rgb_frames, mask_frames, input_frames = [], [], []
            for frame in chunk['reference_frames']:
                index = frame['reference_id']
                rgb = read_reference(frame['source_path'], size)
                for field, pixels in [('image', rgb), ('mask', white)]:
                    path = output / frame[field]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(path), pixels):
                        raise OSError(f'Cannot write reference frame: {path}')
                k = input_K[index].astype(float).copy()
                k[0] *= size[0] / input_size[0]
                k[1] *= size[1] / input_size[1]
                input_frames.append(dict(input_frame_index=len(input_frames), role='reference', **frame,
                                         c2w=input_c2w[index].tolist(), w2c=np.linalg.inv(input_c2w[index]).tolist(),
                                         K=k.tolist(), image_size=size))
                rgb_frames.append(rgb)
                mask_frames.append(white)
            coverage = []
            for target_id in chunk['target_ids']:
                name = f'target_{target_id:06d}.png'
                rgb, mask = cv2.imread(str(rgb_dir / name)), cv2.imread(str(mask_dir / name))
                coverage.append(float(np.mean(mask[:, :, 0] > 0)))
                input_frames.append(dict(input_frame_index=len(input_frames), role='target', target_id=target_id,
                                         image=str((rgb_dir / name).relative_to(root)) if output == root else f'../renders/rgb/{name}',
                                         mask=str((mask_dir / name).relative_to(root)) if output == root else f'../renders/mask/{name}',
                                         **cameras[target_id]))
                rgb_frames.append(rgb)
                mask_frames.append(mask)
            verified_video(output / 'condition' / chunk['video'], rgb_frames, save_video)
            verified_video(output / 'mask' / chunk['video'], mask_frames, save_video)
            (output / 'condition' / (Path(chunk['video']).stem + '_info.txt')).write_text(
                f'Condition Frame Count: {len(chunk["reference_ids"])}\n'
                f'Render Frame Count: {len(chunk["target_ids"])}\n'
                f'Condition Frame Indices: {chunk["reference_ids"]}\n')
            context = 2 if args.context_chunks and chunk['chunk_id'] > 0 else 0
            dump_json(output / chunk['camera_metadata'], dict(scene_id=identity, frames=input_frames,
                      condition_frame_count=len(chunk['reference_ids']), render_frame_count=len(chunk['target_ids']),
                      generated_context_frame_count=context, expected_generation_frames=len(input_frames) + context,
                      reference_pixel_transform='Resize full reference RGB to output size; K scales from reconstruction size'))
            dump_json(output / 'status' / f'chunk_{chunk["chunk_id"]:03d}.json',
                      dict(target_ids=chunk['target_ids'], reference_ids=chunk['reference_ids'],
                           mask_coverage=coverage, reference_method=method, render_device=args.render_device))
            print(f'Prepared {method}/{chunk["video"]}; mask coverage={coverage}', flush=True)
        with (output / 'metadata.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['file_name', 'text', 'mask_index'])
            writer.writeheader()
            for chunk in manifests[method]['chunks']:
                if chunk_complete(output, chunk, size, root, args.render_device):
                    writer.writerow(dict(file_name=chunk['video'], text='', mask_index=','.join(map(str, range(len(chunk['reference_ids']))))))
        command = [sys.executable, 'run_AnyRecon.py', '--root_dir', str(output), '--output_dir', str(output / 'generated'),
                   '--lora_path', 'AnyRecon_full_attention.ckpt', '--width', str(size[0]), '--height', str(size[1]),
                   '--context_chunks' if args.context_chunks else '--independent_chunks', '--export_target_frames']
        if selected_chunks != list(range(len(target_groups))):
            command += ['--chunk_ids'] + list(map(str, sorted(selected_chunks)))
        (output / 'generation_command.txt').write_text(shlex.join(command) + '\n')
        print(f'AnyRecon input ready ({method}): {output}\n{shlex.join(command)}', flush=True)


if __name__ == '__main__':
    main()
