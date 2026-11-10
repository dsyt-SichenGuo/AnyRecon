from contextlib import nullcontext
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

import run_pi3 as original
import run_pi3_reconstruct as reconstruction
from pipeline.reconstruction_visualization import camera_frustum, write_scene_viewer
import test_reconstruction_export as export_tests
from pipeline.reconstruction_io import save_reconstruction


class ReferenceReconstructionTests(unittest.TestCase):
    def test_cloud_matches_original_reference_only_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'refs'
            folder.mkdir()
            for i in range(2):
                y, x = np.indices((84, 112))
                rgb = np.stack([(x + i * 20) % 255, (y * 2) % 255, (x + y) % 255], axis=-1).astype(np.uint8)
                self.assertTrue(cv2.imwrite(str(folder / f'frame_{i:06d}.png'), rgb))
            xyz = torch.zeros(1, 2, 84, 112, 3)
            xyz[..., 0] = torch.arange(112)[None, None, None, :] / 112
            xyz[..., 1] = torch.arange(84)[None, None, :, None] / 84
            xyz[..., 2] = 2
            xyz[:, :, :, 56:, 2] = 4  # Exercise the original depth-edge filter.
            confidence = torch.ones(1, 2, 84, 112, 1)
            confidence[:, :, :2] = -10  # Exercise the confidence threshold.
            poses = torch.eye(4)[None, None].repeat(1, 2, 1, 1)
            poses[0, 1, 0, 3] = 0.4
            result = dict(points=xyz, local_points=xyz, conf=confidence, camera_poses=poses)
            focal = torch.ones(1, 2)
            captured = {}
            original_inputs = []

            def model(images):
                original_inputs.append(images.clone())
                return {k: v.clone() for k, v in result.items()}

            def capture_render(**kwargs):
                captured.update(kwargs)
                empty = np.zeros((84, 112, 3), dtype=np.uint8)
                return [empty] * 2, [empty] * 2, [{}] * 2

            args = SimpleNamespace(num_cond_frames=2, use_retrieval=False,
                                   camera_dir=str(root / 'old_cameras'))
            with patch.object(original, '_compute_target_resolution', return_value=(112, 84)), \
                    patch.object(torch.cuda, 'get_device_capability', return_value=(8, 0)), \
                    patch.object(torch.amp, 'autocast', side_effect=lambda *a, **k: nullcontext()), \
                    patch.object(original, 'recover_focal_shift', return_value=(focal, torch.zeros_like(focal))), \
                    patch.object(original, 'render_point_cloud_to_memory', side_effect=capture_render):
                success, _ = original.process_scene(model, torch.device('cpu'), str(folder), args)
            self.assertTrue(success)
            paths = reconstruction.select_reference_paths(folder)
            images, _, _ = reconstruction.load_reference_images(paths, pixel_limit=9408)
            with patch.object(reconstruction, 'recover_focal_shift', return_value=(focal, torch.zeros_like(focal))):
                new = reconstruction.reconstruct_references(model, images, 'cpu')
            torch.testing.assert_close(original_inputs[0], original_inputs[1], rtol=0, atol=0)
            for old_key, new_key in [('points_3d', 'points'), ('points_rgb', 'colors'),
                                     ('points_source_indices', 'source_indices'),
                                     ('intrinsics', 'intrinsics'), ('extrinsics', 'render_w2c')]:
                np.testing.assert_allclose(captured[old_key], new[new_key], rtol=0, atol=0)

    def test_resize_matches_original_defaults(self):
        for size in [(1280, 960), (1920, 1080), (960, 1280), (512, 512)]:
            fake_image = SimpleNamespace(size=size)
            self.assertEqual(reconstruction.target_resolution(size), original._compute_target_resolution(fake_image))

    def test_camera_frustum_reprojects_to_image_corners(self):
        pose = np.eye(4)
        pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        pose[:3, 3] = [2, 3, 4]
        K = np.array([[6, 0, 1.5], [0, 6, 1], [0, 0, 1]])
        vertices = camera_frustum(pose, K, (4, 3), 0.5)
        np.testing.assert_array_equal(vertices[0], pose[:3, 3])
        camera = (np.linalg.inv(pose) @ np.column_stack([vertices[1:], np.ones(4)]).T).T[:, :3]
        pixels = (K @ camera.T).T
        pixels /= pixels[:, 2:]
        np.testing.assert_allclose(pixels[:, :2], [[0, 0], [3, 0], [3, 2], [0, 2]], atol=1e-12)
        self.assertTrue((camera[:, 2] > 0).all())

    def test_viewer_is_offline_and_preserves_camera_labels(self):
        class Scripts(HTMLParser):
            def __init__(self):
                super().__init__()
                self.external = []

            def handle_starttag(self, tag, attrs):
                if tag == 'script' and 'src' in dict(attrs):
                    self.external.append(dict(attrs)['src'])

        with tempfile.TemporaryDirectory() as directory:
            data = export_tests.ReconstructionExportTests().inputs()
            output = save_reconstruction(directory, **data)
            html_path = write_scene_viewer(output / 'scene.npz', output / 'viewer.html', max_points=2)
            html = html_path.read_text()
            parser = Scripts()
            parser.feed(html)
            self.assertEqual(parser.external, [])
            self.assertIn('Plotly.newPlot', html)
            self.assertIn('参考图 0', html)
            self.assertIn('ref0.png', html)
            self.assertIn('完整点云 4 点', html)
            self.assertIn('当前显示 2 点', html)


if __name__ == '__main__':
    unittest.main()
