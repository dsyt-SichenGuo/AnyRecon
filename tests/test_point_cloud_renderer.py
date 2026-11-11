import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pipeline.point_cloud_renderer import PointCloudRenderer, TensorSplatRenderer
from scripts.prepare_selected_targets import render_backend_matches, chunk_complete


class PointCloudRendererTests(unittest.TestCase):
    def camera(self):
        return dict(w2c=np.eye(4).tolist(), K=[[40, 0, 32], [0, 42, 24], [0, 0, 1]], image_size=[64, 48])

    def assert_matches_original(self, points, colors, camera=None):
        camera = self.camera() if camera is None else camera
        expected, expected_mask = PointCloudRenderer(points, colors).render(camera)
        actual, mask = TensorSplatRenderer(points, colors, 'cpu', batch_size=17).render(camera)
        # CPU NumPy/Numba accumulates sequentially in float32; tensor path uses
        # float64 segmented products/sums. A one-level quantization difference is allowed.
        np.testing.assert_allclose(actual.astype(float), expected.astype(float), atol=1)
        np.testing.assert_array_equal(mask, expected_mask)
        return actual, mask

    def test_depth_ordered_alpha_over_and_bgr_convention(self):
        points = np.array([[.0125, .01, 1], [.025, .02, 2], [.0375, .03, 3]], np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8)
        rgb, _ = self.assert_matches_original(points, colors)
        self.assertGreater(rgb[24, 32, 2], rgb[24, 32, 0])  # nearest point is red, BGR output

    def test_random_cloud_multiple_views_and_projection_batch_invariance(self):
        rng = np.random.default_rng(28)
        points = rng.uniform([-1.5, -1, .15], [1.5, 1, 5], (500, 3)).astype(np.float32)
        colors = rng.integers(1, 256, (500, 3), dtype=np.uint8)
        for offset in [0, .25]:
            camera = self.camera()
            camera['w2c'][0][3] = offset
            actual, mask = self.assert_matches_original(points, colors, camera)
            other, other_mask = TensorSplatRenderer(points, colors, 'cpu', batch_size=500).render(camera)
            np.testing.assert_array_equal(actual, other)
            np.testing.assert_array_equal(mask, other_mask)

    def test_clipping_black_points_opaque_and_empty_views(self):
        points = np.array([[0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, .01],
                           [0, 0, -1], [100, 0, 1], [-.8, -.5, 1]], np.float32)
        colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 0], [255, 255, 0],
                           [0, 255, 255], [255, 0, 255], [128, 128, 128]], np.uint8)
        self.assert_matches_original(points, colors)
        camera = self.camera()
        camera['w2c'][2][3] = -100
        rgb, mask = self.assert_matches_original(points, colors, camera)
        self.assertFalse(rgb.any())
        self.assertFalse(mask.any())

    def test_normalized_float_colors(self):
        self.assert_matches_original(np.array([[0, 0, 2], [.02, .01, 1]], np.float32),
                                     np.array([[.9, .3, .1], [.2, .8, .4]], np.float32))

    def test_backend_cache_records_and_packaging_status_are_both_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(render_backend_matches(root, 0, 'cpu'))
            self.assertFalse(render_backend_matches(root, 0, 'cuda'))
            metadata = root / 'renders/metadata/target_000000.json'
            metadata.parent.mkdir(parents=True)
            metadata.write_text(json.dumps(dict(backend='cuda_alpha_over_v1')))
            self.assertTrue(render_backend_matches(root, 0, 'cuda'))
            self.assertFalse(render_backend_matches(root, 0, 'cpu'))
            # All GPU target PNGs may be ready while old CPU videos still exist.
            # Their CPU completion marker must not let stale videos skip repackaging.
            chunk = dict(chunk_id=0, camera_metadata='cameras/chunk_000.json', video='test.mp4',
                         reference_frames=[], reference_ids=[], target_ids=[0])
            (root/'status').mkdir()
            (root/'status/chunk_000.json').write_text(json.dumps(dict(render_device='cpu')))
            self.assertFalse(chunk_complete(root, chunk, [64, 48], root, 'cuda'))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA hardware unavailable')
    def test_real_cuda_matches_tensor_cpu(self):
        rng = np.random.default_rng(42)
        points = rng.uniform([-.6, -.4, .5], [.6, .4, 3], (500, 3)).astype(np.float32)
        colors = rng.integers(1, 256, (500, 3), dtype=np.uint8)
        expected, expected_mask = TensorSplatRenderer(points, colors, 'cpu', 101).render(self.camera())
        renderer = PointCloudRenderer(points, colors, 'cuda', 101)
        actual, mask = renderer.render(self.camera())
        renderer.close()
        np.testing.assert_allclose(actual.astype(float), expected.astype(float), atol=1)
        np.testing.assert_array_equal(mask, expected_mask)


if __name__ == '__main__':
    unittest.main()
