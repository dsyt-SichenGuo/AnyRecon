import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData

from pipeline.reconstruction_io import save_reconstruction


class ReconstructionExportTests(unittest.TestCase):
    def inputs(self):
        poses = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        poses[:, 0, 3] = [0, 2, 4]
        rendered = poses.copy()
        rendered[2, 0, 3] = 5
        return dict(
            points=np.array([[0, 0, 2], [1, 0, 2], [2, 1, 3], [1, 1, 2]], dtype=np.float32),
            colors=np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255], [12, 34, 56]], dtype=np.uint8),
            source_indices=np.array([0, 1, 0, 1]), input_c2w=poses,
            render_w2c=np.linalg.inv(rendered),
            intrinsics=np.repeat(np.array([[[6, 0, 1.5], [0, 6, 1], [0, 0, 1]]], dtype=np.float32), 3, axis=0),
            reference_rgb=np.full((2, 3, 4, 3), 127, dtype=np.uint8),
            frame_sources=['ref0.png', 'ref1.png', 'target0.png'],
            original_sizes=[(8, 6)] * 3, num_reference_frames=2,
            image_size=(4, 3), output_size=(8, 4), intrinsics_estimated=True,
            settings={'seed': 42}, preview_points=2,
        )

    def test_roundtrip_preserves_cloud_cameras_and_source_mapping(self):
        data = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            output = save_reconstruction(directory, **data)
            with np.load(output / 'scene.npz', allow_pickle=False) as cache:
                for name in ['points', 'colors', 'source_indices', 'input_c2w', 'render_w2c', 'reference_rgb']:
                    np.testing.assert_array_equal(cache[name], data[name])
                np.testing.assert_allclose(cache['input_w2c'] @ cache['input_c2w'], np.tile(np.eye(4), (3, 1, 1)))
                self.assertEqual(cache['render_c2w'][2, 0, 3], 5)
                self.assertEqual(cache['input_c2w'][2, 0, 3], 4)
                self.assertEqual(cache['frame_sources'].tolist(), data['frame_sources'])
            ply = PlyData.read(str(output / 'reference_points.ply'))['vertex'].data
            np.testing.assert_array_equal(np.column_stack([ply[n] for n in ['x', 'y', 'z']]), data['points'])
            np.testing.assert_array_equal(np.column_stack([ply[n] for n in ['red', 'green', 'blue']]), data['colors'])
            self.assertEqual(len(PlyData.read(str(output / 'preview_points.ply'))['vertex'].data), 2)
            centers = PlyData.read(str(output / 'camera_centers.ply'))['vertex'].data
            np.testing.assert_array_equal(centers['x'], [0, 2, 5])
            cameras = json.loads((output / 'cameras.json').read_text())
            self.assertEqual([c['is_reference'] for c in cameras], [True, True, False])
            self.assertEqual(cameras[2]['c2w_render'][0][3], 5)
            self.assertFalse('meters' == json.loads((output / 'metadata.json').read_text())['units'])

    def test_intrinsics_follow_output_resize_and_crop(self):
        data = self.inputs()
        with tempfile.TemporaryDirectory() as directory:
            output = save_reconstruction(directory, **data)
            with np.load(output / 'scene.npz') as cache:
                # 4x3 -> 8x6 resize, then crop one row above and below.
                np.testing.assert_allclose(cache['intrinsics_output'][0], [[12, 0, 3], [0, 12, 1], [0, 0, 1]])
                xyz = np.array([0.2, 0.1, 2])
                before = data['intrinsics'][0] @ xyz
                before /= before[2]
                after = cache['intrinsics_output'][0] @ xyz
                after /= after[2]
                np.testing.assert_allclose(after, [before[0] * 2, before[1] * 2 - 1, 1])

    def test_target_points_are_rejected(self):
        data = self.inputs()
        data['source_indices'][0] = 2
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'only to reference'):
                save_reconstruction(directory, **data)

    def test_reference_only_new_entry_exports_without_targets(self):
        import cv2
        import torch
        import run_pi3_reconstruct as reconstruction

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'reference'
            images.mkdir()
            for index in range(2):
                self.assertTrue(cv2.imwrite(str(images / f'frame_{index:06d}.png'),
                                           np.full((84, 112, 3), 60 + index, dtype=np.uint8)))
            xyz = torch.zeros(1, 2, 84, 112, 3)
            xyz[..., 2] = 2
            poses = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
            poses[0, 1, 0, 3] = 1
            result = dict(points=xyz, local_points=xyz, conf=torch.ones(1, 2, 84, 112, 1), camera_poses=poses)
            paths = reconstruction.select_reference_paths(images)
            with patch.object(reconstruction, 'recover_focal_shift', side_effect=RuntimeError('test fallback')):
                output = reconstruction.reconstruct_and_export(
                    lambda x: result, paths, root / 'output', device='cpu',
                    pixel_limit=9408, preview_points=100, viewer=False,
                )
            self.assertTrue((output / 'reference_points.ply').is_file())
            with np.load(output / 'scene.npz', allow_pickle=False) as cache:
                self.assertEqual(int(cache['num_reference_frames']), 2)
                self.assertEqual(cache['input_c2w'].shape, (2, 4, 4))
                self.assertEqual(cache['reference_rgb'].shape, (2, 84, 112, 3))
            self.assertFalse(json.loads((output / 'metadata.json').read_text())['intrinsics_estimated'])
            self.assertEqual(json.loads((output / 'metadata.json').read_text())['num_target_frames'], 0)


if __name__ == '__main__':
    unittest.main()
