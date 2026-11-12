"""Select straight-advance + horizontal panorama chunks in the Pi3 world frame."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.target_cameras import TargetCameraStore, scene_identity


def load_viewer_scene(path, max_points):
    if max_points < 0:
        raise ValueError('max_points must be >= 0; 0 displays all points')
    with np.load(path, allow_pickle=False) as scene:
        poses = scene['input_c2w'].astype(float)
        K = scene['intrinsics_reconstruction'].astype(float)
        size = scene['image_size_reconstruction'].tolist()
        sources = scene['frame_sources'].tolist()
        # Load large compressed members sequentially; do not load reference_rgb.
        points = scene['points']
        total = len(points)
        if not total:
            raise ValueError('Empty scene point cloud')
        ids = np.linspace(0, total - 1, min(total, max_points or total), dtype=np.int64)
        sample = points[ids].astype('<f4')
        del points
        colors = scene['colors']
        rgb = colors[ids].astype('<f4') / 255
        del colors
    if not np.isfinite(sample).all():
        raise ValueError('Point cloud contains non-finite coordinates')
    extent = np.percentile(sample, 98, axis=0) - np.percentile(sample, 2, axis=0)
    span = max(float(np.linalg.norm(extent)), 1e-4)
    config = dict(poses=poses.tolist(), intrinsics=K.tolist(), image_size=size,
                  sources=sources, total_points=total, shown_points=len(sample), span=span,
                  scene_path=str(Path(path).resolve()))
    cloud = np.concatenate([sample, rgb], axis=1).astype('<f4').tobytes()
    return config, cloud, scene_identity(poses, K, size)


def make_handler(config, cloud, store, token):
    config = dict(config, output_dir=str(store.root.resolve()))
    page = (Path(__file__).with_name('target_camera_viewer.html').read_text()
            .replace('__SAVE_TOKEN__', token)).encode()

    class Handler(BaseHTTPRequestHandler):
        def reply(self, content, content_type='application/json', status=200):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(content)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                self.reply(page, 'text/html; charset=utf-8')
            elif self.path == '/api/config':
                self.reply(json.dumps(config).encode())
            elif self.path == '/api/points':
                self.reply(cloud, 'application/octet-stream')
            elif self.path == '/api/targets':
                self.reply(json.dumps(store.document()).encode())
            else:
                self.reply(b'{"error":"Not found"}', status=404)

        def do_POST(self):
            if self.path not in ('/api/targets', '/api/chunks', '/api/undo', '/api/import') or self.headers.get('X-Save-Token') != token:
                self.reply(b'{"error":"Invalid endpoint or token"}', status=403)
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= (32 * 1024 * 1024 if self.path in ('/api/chunks', '/api/import') else 16384):
                    raise ValueError('Invalid request length')
                payload = json.loads(self.rfile.read(size))
                if self.path == '/api/import':
                    store.import_document(payload)
                    result = dict(frame_count=len(store.cameras), chunk_count=len(store.chunks))
                elif self.path == '/api/chunks':
                    result = dict(chunk=store.append_chunk(payload))
                elif self.path == '/api/undo':
                    result = dict(restore_camera=store.undo())
                else:
                    result = dict(frame_id=store.append(payload))
                self.reply(json.dumps(dict(result, output_dir=str(store.root.resolve()))).encode())
            except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
                self.reply(json.dumps(dict(error=str(error))).encode(), status=400)
            except OSError as error:
                self.reply(json.dumps(dict(error=f'Failed to write output: {error}')).encode(), status=500)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--load_targets', type=Path, help='Load target_cameras.json or its directory into an empty output; omit to resume output_dir automatically')
    parser.add_argument('--max_points', type=int, default=1000000, help='Display cap only; 0 = full cloud')
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    print('Loading scene on CPU; no Pi3 or AnyRecon models are loaded.', flush=True)
    config, cloud, scene_id = load_viewer_scene(args.scene, args.max_points)
    store = TargetCameraStore(args.output_dir, args.scene, scene_id)
    if args.load_targets is not None:
        source = args.load_targets
        if source.is_dir():
            source = source / 'target_cameras.json'
        try:
            store.import_document(json.loads(source.read_text()))
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f'Cannot load target cameras: {error}')
    print(f'Resuming {len(store.chunks)} chunks / {len(store.cameras)} target cameras.', flush=True)
    server = HTTPServer(('127.0.0.1', args.port), make_handler(config, cloud, store, secrets.token_hex(24)))
    print(f'Displaying {config["shown_points"]:,}/{config["total_points"]:,} points', flush=True)
    print(f'Open http://127.0.0.1:{args.port} — targets saved to {store.root.resolve()}', flush=True)
    print('Remote machine: forward this port with ssh -L PORT:127.0.0.1:PORT USER@HOST', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
