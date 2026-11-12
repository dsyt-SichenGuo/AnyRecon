import copy
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.target_cameras import TargetCameraStore, sample_advance_panorama


class AdvancePanoramaTests(unittest.TestCase):
    def payload(self):
        start = dict(c2w=np.eye(4).tolist(), K=[[80, 0, 32], [0, 90, 24], [0, 0, 1]],
                     image_size=[64, 48])
        end = copy.deepcopy(start)
        end['c2w'][0][3] = 4
        return dict(kind='advance_panorama', recording_id='first', camera=end,
                    start_camera=start, advance_frames=5, panorama_frames=25, world_up=[0, -1, 0])

    def test_equal_distance_advance_and_full_uniform_yaw(self):
        cameras, distance, samples, start, metadata = sample_advance_panorama(self.payload())
        self.assertEqual(len(cameras), 30)
        poses = np.asarray([c['c2w'] for c in cameras])
        np.testing.assert_allclose(poses[:5, 0, 3], [0, 1, 2, 3, 4])
        np.testing.assert_allclose(poses[5:, :3, 3], np.tile([4, 0, 0], (25, 1)))
        np.testing.assert_allclose(poses[4:, :3, 1], np.tile([0, 1, 0], (26, 1)))
        relative = np.transpose(poses[4:-1, :3, :3], (0, 2, 1)) @ poses[5:, :3, :3]
        np.testing.assert_allclose(Rotation.from_matrix(relative).magnitude(), 2 * np.pi / 25)
        np.testing.assert_allclose(poses[-1], poses[4])
        self.assertEqual(distance, 4)
        self.assertEqual(samples, [0, 1, 2, 3, 4] + [4] * 25)
        self.assertEqual(metadata['panorama_angles_degrees'][-1], 360)

    def test_tilted_world_up_and_level_destination(self):
        payload = self.payload()
        change = Rotation.from_euler('xyz', [0.4, 0.2, -0.3]).as_matrix()
        pose = np.asarray(payload['camera']['c2w'])
        pose[:3, :3] = change @ Rotation.from_euler('xyz', [0.2, 0, 0.3]).as_matrix()
        payload['camera']['c2w'] = pose.tolist()
        payload['world_up'] = change @ np.array([0, -1, 0])
        cameras, *_ = sample_advance_panorama(payload)
        poses = np.asarray([c['c2w'] for c in cameras])
        np.testing.assert_allclose(poses[4:, :3, 1], np.tile(-payload['world_up'], (26, 1)), atol=1e-12)
        np.testing.assert_allclose(poses[4:, :3, 2] @ payload['world_up'], 0, atol=1e-12)

    def test_authoritative_previous_endpoint_persistence_retry_and_undo(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'world')
            first = self.payload()
            store.append_chunk(first)
            endpoint = copy.deepcopy(store.cameras[-1])
            second = self.payload()
            second.update(recording_id='second', advance_frames=3, panorama_frames=8)
            second['camera']['c2w'][0][3] = 8
            second['start_camera']['c2w'][0][3] = -100  # Ignore stale client state.
            chunk = store.append_chunk(second)
            self.assertEqual(chunk['advance_target_ids'], [30, 31, 32])
            self.assertEqual(chunk['panorama_target_ids'], list(range(33, 41)))
            np.testing.assert_allclose(store.cameras[30]['c2w'], endpoint['c2w'])
            self.assertEqual(store.append_chunk(second), chunk)
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'world')
            self.assertEqual(len(store.cameras), 41)
            self.assertEqual(store.undo(), endpoint)
            self.assertEqual(len(store.cameras), 30)
            self.assertEqual(store.undo()['c2w'], first['start_camera']['c2w'])

    def test_stationary_panorama_single_advance_and_rejections(self):
        payload = self.payload()
        payload['camera'] = copy.deepcopy(payload['start_camera'])
        payload['advance_frames'] = 1
        self.assertEqual(len(sample_advance_panorama(payload)[0]), 26)
        for changes in [dict(advance_frames=0), dict(panorama_frames=2),
                        dict(advance_frames=True), dict(panorama_frames=1000),
                        dict(world_up=[0, 0, 0]), dict(world_up=[0, 0, 1])]:
            with self.assertRaises(ValueError):
                sample_advance_panorama(dict(payload, **changes))


if __name__ == '__main__':
    unittest.main()
