from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from pipeline.target_cameras import scene_identity
from scripts.prepare_selected_targets import reference_target_groups, reuse_target_renders


class ReferenceTargetScopeTests(unittest.TestCase):
    def test_variable_segment_lengths_and_legacy_fallback(self):
        groups = [list(range(30)), list(range(30, 41)), [41, 42]]
        document = dict(chunks=[dict(kind='advance_panorama', advance_frames=5, panorama_frames=25),
                                dict(kind='advance_panorama', advance_frames=3, panorama_frames=8),
                                dict(kind='recorded')])
        self.assertEqual(reference_target_groups(document, groups, 'panorama'),
                         [list(range(5, 30)), list(range(33, 41)), [41, 42]])
        self.assertEqual(reference_target_groups(document, groups, 'all'), groups)
        self.assertEqual(reference_target_groups({}, groups, 'panorama'), groups)
        bad = copy.deepcopy(document)
        bad['chunks'][0]['panorama_target_ids'] = [0]
        with self.assertRaises(ValueError):
            reference_target_groups(bad, groups, 'panorama')

    def test_scope_changes_selection_preserves_all_outputs_and_reuses_renders(self):
        # Five repeated forward-facing cameras favor reference 0. The panorama
        # cameras see reference 1. Both methods must switch when forward is excluded.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            front, back = np.eye(4), np.diag([-1., 1., -1., 1.])
            poses = np.stack([front, back])
            K = np.repeat(np.array([[[30., 0, 32], [0, 32, 24], [0, 0, 1]]]), 2, axis=0)
            paths = []
            for i in range(2):
                path = root / f'ref_{i}.png'
                Image.new('RGB', (64, 48), (i * 200, 80, 100)).save(path)
                paths.append(str(path))
            points = np.array([[x, 0, 2] for x in np.linspace(-.6, .6, 8)] +
                              [[x, 0, -2] for x in np.linspace(-.6, .6, 4)], dtype=np.float32)
            np.savez(root / 'scene.npz', points=points, colors=np.full((12, 3), 150, np.uint8),
                     source_indices=np.array([0] * 8 + [1] * 4), input_c2w=poses,
                     intrinsics_reconstruction=K, image_size_reconstruction=[64, 48],
                     num_reference_frames=2, frame_sources=paths)
            cameras = [dict(c2w=p.tolist(), K=K[0].tolist(), image_size=[64, 48]) for p in [front]*5 + [back]*3]
            document = dict(scene_id=scene_identity(poses, K, [64, 48]), cameras=cameras,
                            chunks=[dict(kind='advance_panorama', advance_frames=5, panorama_frames=3,
                                         target_ids=list(range(8)))])
            (root / 'cameras.json').write_text(json.dumps(document))
            entry = Path(__file__).resolve().parents[1] / 'scripts/prepare_selected_targets.py'
            argv = [str(entry), '--scene', str(root / 'scene.npz'), '--cameras', str(root / 'cameras.json'),
                    '--output_dir', str(root / 'all'), '--reference_method', 'compare',
                    '--num_reference', '1', '--retrieval_width', '32', '--context_chunks']
            with patch.object(sys, 'argv', argv + ['--reference_target_scope', 'all']), redirect_stdout(io.StringIO()):
                runpy.run_path(str(entry), run_name='__main__')
            for method in ['original', 'improved']:
                m = json.loads((root / 'all' / method / 'manifest.json').read_text())
                self.assertEqual(m['chunks'][0]['reference_ids'], [0])
            panorama_argv = argv.copy()
            panorama_argv[panorama_argv.index('--output_dir')+1] = str(root / 'panorama')
            panorama_argv += ['--reuse_renders_from', str(root / 'all')]
            with patch.object(sys, 'argv', panorama_argv), redirect_stdout(io.StringIO()), \
                    patch('pipeline.point_cloud_renderer.PointCloudRenderer') as renderer:
                runpy.run_path(str(entry), run_name='__main__')
            renderer.assert_not_called()
            self.assertEqual(len(list((root / 'panorama/renders/rgb').glob('*.png'))), 8)
            for method in ['original', 'improved']:
                m = json.loads((root / 'panorama' / method / 'manifest.json').read_text())
                chunk = m['chunks'][0]
                self.assertEqual(chunk['reference_ids'], [1])
                self.assertEqual(chunk['target_ids'], list(range(8)))
                self.assertEqual(chunk['retrieval']['retrieval_target_ids'], [5, 6, 7])
                self.assertEqual(chunk['retrieval']['excluded_target_ids'], list(range(5)))
                self.assertEqual(len(chunk['retrieval']['selected_coverage_fraction_per_target']), 3)
                cap = cv2.VideoCapture(str(root / 'panorama' / method / 'condition' / chunk['video']))
                self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 9)  # 1 reference + all 8 targets.
                cap.release()
            config = json.loads((root / 'panorama/preparation_config.json').read_text())
            self.assertEqual(config['reference_target_scope'], 'panorama')
            config['cameras_sha256'] = 'another-trajectory'
            with self.assertRaisesRegex(ValueError, 'incompatible cameras_sha256'):
                reuse_target_renders(root / 'all', root / 'rejected', config, [64, 48], 'cpu', [0])


if __name__ == '__main__':
    unittest.main()
