from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from pipeline.cuda_reference_selection import CudaCoverage, CudaReferenceRetriever
from pipeline.reference_selection import original_source_counts, visible_source_coverage
from scripts.prepare_selected_targets import select_references
from pipeline.target_cameras import scene_identity


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
class CudaReferenceTests(unittest.TestCase):
    def camera(self):
        return dict(w2c=np.eye(4).tolist(), K=[[30, 0, 32], [0, 32, 24], [0, 0, 1]], image_size=[64, 48])

    def test_projection_counts_visibility_and_candidate_occlusion(self):
        rng = np.random.default_rng(9)
        points = rng.uniform([-3, -3, -1], [3, 3, 6], (10003, 3)).astype(np.float32)
        colors = np.full((len(points), 3), 127, np.uint8)
        sources = rng.integers(0, 8, len(points), dtype=np.int32)
        colors[::10] = 0
        sample = points[colors.any(axis=1)]
        sample_sources = sources[colors.any(axis=1)]
        pool = [1, 4, 7]
        gpu = CudaReferenceRetriever(sample, sample_sources, 8, pool, points, colors, sources, batch_size=731)
        camera = self.camera()
        expected = original_source_counts(points, colors, sources, camera, 8)[pool]
        np.testing.assert_array_equal(gpu.original_counts(camera), expected)
        expected = visible_source_coverage(sample, sample_sources, camera, 8, width=32, candidate_ids=pool)
        np.testing.assert_array_equal(gpu.coverage(camera, width=32).cpu().numpy(), expected)
        gpu = CudaReferenceRetriever(np.array([[0, 0, 1], [0, 0, 2]], np.float32),
                                     np.array([0, 1]), 2, [1])
        self.assertFalse(gpu.coverage(camera).any().item())

    def test_both_selection_methods_and_all_diagnostics_match_cpu(self):
        rng = np.random.default_rng(11)
        for targets, references, count in [(30, 40, 15), (3, 9, 6), (1, 5, 3)]:
            masks = rng.random((targets, references, 193)) < .08
            masks[:, 1] = masks[:, 0]  # Include tied/redundant candidates.
            counts = rng.integers(0, 100, (targets, references))
            ids = list(range(0, references * 2, 2))
            gpu = CudaCoverage(torch.tensor(masks, device='cuda'))
            for method in ['original', 'improved']:
                a, da = select_references(method, masks, counts, ids, count, 10)
                b, db = select_references(method, gpu, counts, ids, count, 10)
                self.assertEqual(a, b)
                self.assertEqual(da, db)
        masks = np.ones((3, 8, 16), bool)  # All gain scores tied.
        a = select_references('improved', masks, None, list(range(8)), 2, 10)
        b = select_references('improved', CudaCoverage(torch.tensor(masks, device='cuda')), None, list(range(8)), 2, 10)
        self.assertEqual(a, b)
        with self.assertRaisesRegex(ValueError, 'no sampled visible'):
            CudaCoverage(torch.zeros((2, 3, 8), dtype=torch.bool, device='cuda')).choose(2)

    def test_cli_compare_cuda_plan_and_partial_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pose = np.eye(4)[None]
            K = np.array([[[30., 0, 32], [0, 32, 24], [0, 0, 1]]])
            paths = []
            for i in range(4):
                p = root / f'ref_{i}.png'
                Image.new('RGB', (64, 48), (80, i * 30, 100)).save(p)
                paths.append(str(p))
            np.savez(root / 'scene.npz', points=np.array([[-.3, 0, 2], [0, 0, 2], [.3, 0, 2], [.6, 0, 2]], np.float32),
                     colors=np.full((4, 3), 100, np.uint8), source_indices=np.arange(4),
                     input_c2w=pose, intrinsics_reconstruction=K, image_size_reconstruction=[64, 48],
                     num_reference_frames=4, frame_sources=paths)
            camera = dict(c2w=pose[0].tolist(), K=K[0].tolist(), image_size=[64, 48])
            (root / 'cameras.json').write_text(json.dumps(dict(
                scene_id=scene_identity(pose, K, [64, 48]), cameras=[camera] * 4)))
            entry = Path(__file__).resolve().parents[1] / 'scripts/prepare_selected_targets.py'
            argv = [str(entry), '--scene', str(root / 'scene.npz'), '--cameras', str(root / 'cameras.json'),
                    '--output_dir', str(root / 'prepared'), '--reference_method', 'compare',
                    '--num_reference', '2', '--chunk_size', '2', '--retrieval_width', '32',
                    '--retrieval_device', 'cuda', '--context_chunks', '--plan_only']
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                runpy.run_path(str(entry), run_name='__main__')
            for method in ['original', 'improved']:
                path = root / 'prepared' / method / 'manifest.json'
                m = json.loads(path.read_text())
                self.assertEqual(len(m['chunks']), 2)
                self.assertFalse(m['independent_chunks'])
                self.assertEqual(m['chunks'][0]['retrieval']['retrieval_backend'], 'cuda')
                m['chunks'] = m['chunks'][:1]
                path.write_text(json.dumps(m))
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()) as output:
                runpy.run_path(str(entry), run_name='__main__')
            self.assertNotIn('Retrieving chunk 1/2', output.getvalue())
            self.assertIn('Retrieving chunk 2/2', output.getvalue())


if __name__ == '__main__':
    unittest.main()
