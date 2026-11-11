import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from pipeline.target_cameras import TargetCameraStore, sample_trajectory, target_chunk_groups


class TrajectoryRecordingTests(unittest.TestCase):
    def recording(self, frames=5, offset=0):
        poses = np.tile(np.eye(4), (3, 1, 1))
        poses[:, :3, :3] = Rotation.from_euler('z', [0, 90, 180], degrees=True).as_matrix()
        poses[:, 0, 3] = np.array([0, 2, 8]) + offset
        camera = dict(c2w=poses[0].tolist(), K=[[80, 0, 32], [0, 90, 24], [0, 0, 1]], image_size=[64, 48])
        return dict(camera=camera, frame_count=frames,
                    samples=[dict(c2w=p.tolist()) for p in poses])

    def test_distance_sampling_interpolates_rotations_and_preserves_endpoints(self):
        recording = self.recording(9)
        cameras, length, distances, restore = sample_trajectory(recording)
        self.assertEqual(length, 8)
        self.assertEqual(distances, np.linspace(0, 8, 9).tolist())
        np.testing.assert_allclose(np.asarray([c['c2w'] for c in cameras])[:, 0, 3], np.arange(9))
        np.testing.assert_allclose(cameras[1]['c2w'],
                                   np.block([[Rotation.from_euler('z', 45, degrees=True).as_matrix(), np.array([[1], [0], [0]])],
                                             [np.array([[0, 0, 0, 1]])]]), atol=1e-12)
        self.assertEqual(cameras[0]['c2w'], recording['samples'][0]['c2w'])
        self.assertEqual(cameras[-1]['c2w'], recording['samples'][-1]['c2w'])
        for c in cameras:
            np.testing.assert_allclose(np.asarray(c['w2c']) @ c['c2w'], np.eye(4), atol=1e-12)
        self.assertEqual(restore, cameras[0])

    def test_multiple_chunks_restart_undo_and_empty_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'scene')
            first = self.recording(3)
            second = self.recording(7, offset=10)
            store.append_chunk(first)
            store.append_chunk(second)
            self.assertEqual(target_chunk_groups(store.document(), 4), [[0, 1, 2], list(range(3, 10))])
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'scene')
            with np.load(Path(directory)/'target_cameras.npz', allow_pickle=False) as data:
                self.assertEqual(data['chunk_offsets'].tolist(), [0, 3, 10])
                self.assertEqual(data['chunk_ids'].tolist(), [0]*3+[1]*7)
            restored = store.undo()
            self.assertEqual(restored['c2w'], second['camera']['c2w'])
            self.assertEqual(len(store.cameras), 3)
            store.undo()
            with np.load(Path(directory)/'target_cameras.npz', allow_pickle=False) as data:
                self.assertEqual(data['c2w'].shape, (0, 4, 4))
                self.assertEqual(data['chunk_offsets'].tolist(), [0])
            self.assertEqual(json.loads(store.path.read_text())['chunks'], [])
            with self.assertRaisesRegex(ValueError, 'No saved chunk'):
                store.undo()
            chunk = store.append_chunk(first)
            self.assertEqual(chunk['chunk_id'], 0)
            self.assertEqual(chunk['target_ids'], [0, 1, 2])

    def test_invalid_recording_does_not_write_partial_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'scene')
            store.append_chunk(self.recording())
            before = store.path.read_bytes()
            invalids = []
            for count in [0, 1, 1001, 3.2, True]:
                invalids.append(dict(self.recording(), frame_count=count))
            bad = self.recording()
            bad['samples'] = [copy.deepcopy(bad['samples'][0]) for _ in range(3)]
            invalids.append(bad)
            bad = self.recording()
            bad['samples'][1]['c2w'][0][0] = 9
            invalids.append(bad)
            bad = self.recording()
            bad['camera']['image_size'] = [128, 96]
            invalids.append(bad)
            for bad in invalids:
                with self.assertRaises(ValueError):
                    store.append_chunk(bad)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertEqual(len(store.cameras), 5)

    def test_retry_does_not_duplicate_saved_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'scene')
            recording = dict(self.recording(), recording_id='test-recording')
            first = store.append_chunk(recording)
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'scene')
            self.assertEqual(store.append_chunk(recording), first)
            self.assertEqual(len(store.cameras), 5)
            altered = copy.deepcopy(recording)
            altered['frame_count'] = 8
            with self.assertRaisesRegex(ValueError, 'different trajectory'):
                store.append_chunk(altered)

    def test_shortest_rotation_path_and_zero_distance_rejection(self):
        recording = self.recording(3)
        a, b = np.eye(4), np.eye(4)
        a[:3, :3], b[:3, :3] = Rotation.from_euler('z', [170, -170], degrees=True).as_matrix()
        b[0, 3] = 2
        recording['camera']['c2w'] = a.tolist()
        recording['samples'] = [dict(c2w=a.tolist()), dict(c2w=b.tolist())]
        cameras, *_ = sample_trajectory(recording)
        np.testing.assert_allclose(np.asarray(cameras[1]['c2w'])[:3, :3],
                                   Rotation.from_euler('z', 180, degrees=True).as_matrix(), atol=1e-12)
        recording['samples'][1]['c2w'] = a.tolist()
        with self.assertRaisesRegex(ValueError, 'zero or invalid length'):
            sample_trajectory(recording)
        b[0, 3] = 0
        recording['samples'][1]['c2w'] = b.tolist()
        with self.assertRaisesRegex(ValueError, 'zero or invalid length'):
            sample_trajectory(recording)

    def test_pauses_and_sample_density_do_not_change_distance_sampling(self):
        original = self.recording(11)
        dense = copy.deepcopy(original)
        dense['samples'] = ([copy.deepcopy(original['samples'][0])] * 40
                            + [copy.deepcopy(original['samples'][1])] * 3
                            + [copy.deepcopy(original['samples'][2])] * 20)
        first, length, distances, _ = sample_trajectory(original)
        second, dense_length, dense_distances, _ = sample_trajectory(dense)
        self.assertEqual(length, dense_length)
        self.assertEqual(distances, dense_distances)
        np.testing.assert_allclose([c['c2w'] for c in first], [c['c2w'] for c in second])
        with tempfile.TemporaryDirectory() as directory:
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'scene')
            chunk = store.append_chunk(original)
            self.assertEqual(chunk['sampling'], 'position_arc_length')
            self.assertEqual(chunk['path_length'], 8)
            self.assertEqual(chunk['sample_distances'], distances)
            self.assertNotIn('sample_times', chunk)
            self.assertNotIn('duration_seconds', chunk)

    def test_corner_and_closed_loop_use_total_path_not_endpoint_distance(self):
        recording = self.recording(9)
        positions = [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [0, 0, 0]]
        poses = np.tile(np.eye(4), (len(positions), 1, 1))
        poses[:, :3, 3] = positions
        recording['camera']['c2w'] = poses[0].tolist()
        recording['samples'] = [dict(c2w=p.tolist()) for p in poses]
        cameras, length, distances, _ = sample_trajectory(recording)
        self.assertEqual(length, 8)
        np.testing.assert_allclose(np.asarray([c['c2w'] for c in cameras])[:, :3, 3],
                                   [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0],
                                    [1, 2, 0], [0, 2, 0], [0, 1, 0], [0, 0, 0]])

    def test_legacy_migration_and_invalid_chunk_boundaries(self):
        recording = self.recording()
        cameras, *_ = sample_trajectory(recording)
        document = dict(scene_id='scene', cameras=cameras)
        self.assertEqual(target_chunk_groups(document, 3), [[0, 1, 2], [3, 4]])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'target_cameras.json').write_text(json.dumps(document))
            store = TargetCameraStore(directory, '/tmp/scene.npz', 'scene')
            self.assertEqual(len(store.cameras), 5)
            store.append_chunk(recording)
            self.assertEqual([len(c['target_ids']) for c in store.chunks], [5, 5])
        for ids in [[0, 2, 1, 3, 4], [0, 1], [0, 1, 2, 3, 3], [False, 1, 2, 3, 4]]:
            with self.assertRaises(ValueError):
                target_chunk_groups(dict(document, chunks=[dict(target_ids=ids)]), 4)


if __name__ == '__main__':
    unittest.main()
