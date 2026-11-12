import importlib.util
from http.server import HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from pipeline.target_cameras import TargetCameraStore, scene_identity, validate_camera

spec = importlib.util.spec_from_file_location('select_targets', Path(__file__).resolve().parents[1] / 'scripts/select_target_cameras.py')
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class TargetCameraTests(unittest.TestCase):
    def camera(self):
        # An asymmetric translated/rotated camera catches inverse and axis mistakes.
        return dict(c2w=[[0, 0, 1, 2], [0, 1, 0, 3], [-1, 0, 0, 4], [0, 0, 0, 1]],
                    K=[[80, 0, 23], [0, 90, 18], [0, 0, 1]], image_size=[64, 48])

    def test_renderer_pose_convention_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'test-world')
            self.assertEqual(store.append(self.camera()), 0)
            store.append(dict(self.camera(), image_size=[128, 96]))
            restored = TargetCameraStore(directory, '/tmp/scene.npz', 'test-world')
            self.assertEqual(len(restored.cameras), 2)
            with np.load(Path(directory) / 'target_cameras.npz', allow_pickle=False) as data:
                self.assertEqual(data['w2c'].shape, (2, 4, 4))
                np.testing.assert_allclose(data['w2c'] @ data['c2w'], np.broadcast_to(np.eye(4), (2, 4, 4)))
                local = np.array([.1, .2, 2., 1])
                world = data['c2w'][0] @ local
                camera = data['w2c'][0] @ world
                uv = data['K'][0] @ camera[:3]
                np.testing.assert_allclose(uv[:2] / uv[2], [27, 27])
            with self.assertRaisesRegex(ValueError, 'different'):
                TargetCameraStore(directory, '/tmp/other.npz', 'other-world')

    def test_invalid_camera_is_not_persisted(self):
        camera = self.camera()
        for invalid in [dict(camera, image_size=[1260, 960]),
                        dict(camera, K=[[-1, 0, 23], [0, 90, 18], [0, 0, 1]]),
                        dict(camera, c2w=np.diag([1, -1, 1, 1]).tolist()),
                        dict(camera, c2w=np.full((4, 4), np.nan).tolist())]:
            with self.assertRaises(ValueError):
                validate_camera(invalid)

    def test_original_renderer_accepts_saved_camera_without_axis_conversion(self):
        from run_pi3 import render_point_cloud_to_memory
        camera = validate_camera(self.camera())
        local = np.array([.1, .2, 2., 1])
        world = np.asarray(camera['c2w']) @ local
        frames, masks, _ = render_point_cloud_to_memory(
            world[None, :3], np.array([[255, 0, 0]], dtype=np.uint8),
            np.asarray([camera['w2c']]), np.asarray([camera['K']]), (64, 48))
        # Renderer returns BGR, projected red point is at u=27, v=27.
        np.testing.assert_array_equal(frames[0][27, 27], [0, 0, 255])
        self.assertTrue(masks[0][27, 27].all())

    def test_scene_sampling_preserves_world_and_avoids_reference_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scene.npz'
            poses = np.asarray([self.camera()['c2w']])
            intrinsics = np.asarray([self.camera()['K']])
            points = np.arange(30, dtype=np.float32).reshape(10, 3)
            # No reference_rgb member: loading it is unnecessary for free-camera selection.
            np.savez(path, input_c2w=poses, intrinsics_reconstruction=intrinsics,
                     image_size_reconstruction=[64, 48], frame_sources=['ref.png'],
                     points=points, colors=np.full((10, 3), 127, dtype=np.uint8))
            config, binary, identity = viewer.load_viewer_scene(path, 3)
            data = np.frombuffer(binary, dtype='<f4').reshape(-1, 6)
            np.testing.assert_array_equal(data[:, :3], points[[0, 4, 9]])
            self.assertEqual(config['total_points'], 10)
            self.assertEqual(identity, scene_identity(poses, intrinsics, [64, 48]))
            self.assertNotEqual(identity, scene_identity(poses + .01, intrinsics, [64, 48]))

    def test_http_save_requires_token_and_writes_numeric_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'test-world')
            server = HTTPServer(('127.0.0.1', 0), viewer.make_handler({'test': True}, b'cloud', store, 'token'))
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base + '/') as response:
                    page = response.read().decode()
                self.assertIn("const token='token'", page)
                self.assertNotIn('https://', page)
                request = Request(base + '/api/targets', json.dumps(self.camera()).encode(), method='POST')
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 403)
                request.add_header('X-Save-Token', 'token')
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['frame_id'], 0)
                with urlopen(base + '/api/targets') as response:
                    self.assertEqual(len(json.load(response)['cameras']), 1)
                end = self.camera()['c2w']
                end[0][3] += 2
                recording = dict(camera=self.camera(), frame_count=3,
                                 samples=[dict(c2w=self.camera()['c2w']), dict(c2w=end)])
                request = Request(base + '/api/chunks', json.dumps(recording).encode(), method='POST')
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 403)
                request.add_header('X-Save-Token', 'token')
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['chunk']['target_ids'], [1, 2, 3])
                request = Request(base + '/api/undo', b'{}', method='POST')
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 403)
                request.add_header('X-Save-Token', 'token')
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['restore_camera'], validate_camera(self.camera()))
                self.assertEqual(len(store.cameras), 1)
                self.assertEqual(len(store.chunks), 1)
                request = Request(base + '/api/import', json.dumps(store.document()).encode(), method='POST')
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 403)
                request.add_header('X-Save-Token', 'token')
                with urlopen(request) as response:
                    self.assertEqual(json.load(response)['frame_count'], 1)
                wrong = dict(store.document(), scene_id='wrong-world')
                request = Request(base + '/api/import', json.dumps(wrong).encode(), method='POST',
                                  headers={'X-Save-Token': 'token'})
                with self.assertRaises(HTTPError) as error:
                    urlopen(request)
                self.assertEqual(error.exception.code, 400)
                self.assertEqual(len(store.cameras), 1)
            finally:
                server.shutdown()
                server.server_close()
                worker.join()


if __name__ == '__main__':
    unittest.main()
