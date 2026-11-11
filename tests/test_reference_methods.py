import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from pipeline.reference_selection import (original_source_counts, choose_original_references,
                                         visible_source_coverage, choose_chunk_references, coverage_diagnostic)
from scripts.prepare_selected_targets import resolve_reference_pool, select_references


class ReferenceMethodTests(unittest.TestCase):
    def camera(self):
        return dict(w2c=np.eye(4).tolist(), K=[[16, 0, 16], [0, 16, 16], [0, 0, 1]], image_size=[32, 32])

    def test_original_full_counts_match_upstream_renderer_including_occluded_points(self):
        from run_pi3 import render_point_cloud_to_memory
        camera = self.camera()
        points = np.array([[0, 0, 1], [0, 0, 2], [.25, 0, 1], [0, 0, .05],
                           [100, 0, 1], [0, 0, 3], [-1, 0, 1]], dtype=np.float32)
        colors = np.full((len(points), 3), 100, dtype=np.uint8)
        colors[5] = 0
        sources = np.array([0, 1, 2, 0, 2, 2, 1], dtype=np.int32)
        _, _, stats = render_point_cloud_to_memory(points, colors, np.array([camera['w2c']]),
                                                   np.array([camera['K']]), (32, 32), sources)
        counts = original_source_counts(points, colors, sources, camera, 3, block_size=2)
        self.assertEqual(dict(enumerate(counts.tolist())), {i: stats[0].get(i, 0) for i in range(3)})
        self.assertEqual(counts[1], 2)  # Behind source 0 still counts, exactly as upstream.

    def test_original_shortlist_aggregation_and_possible_smaller_actual_count(self):
        counts = np.array([[10, 9, 8], [0, 1, 20]])
        selected, report = choose_original_references(counts, 2, top_per_target=1)
        self.assertEqual(selected, [0, 2])
        self.assertEqual(report['aggregate_source_counts'], {'0': 10, '2': 20})
        selected, _ = choose_original_references(np.ones((1, 20)), 15)
        self.assertEqual(len(selected), 10)

    def test_improved_adds_complementary_pixels_and_reports_original_scene_ids(self):
        masks = np.zeros((1, 3, 12), dtype=bool)
        masks[0, 0, :8] = True
        masks[0, 1, :8] = True
        masks[0, 2, 8:] = True
        counts = np.array([[100, 90, 5]])
        original, a = select_references('original', masks, counts, [4, 9, 13], 2, 10)
        improved, b = select_references('improved', masks, counts, [4, 9, 13], 2, 10)
        self.assertEqual(original, [4, 9])
        self.assertEqual(improved, [4, 13])
        self.assertGreater(b['min_coverage'], a['min_coverage'])
        self.assertEqual(b['best_reference_per_target'], [4])
        self.assertEqual(b['selection_order'], [4, 13])
        self.assertEqual(set(a['aggregate_source_counts']), {'4', '9', '13'})

    def test_excluding_a_reference_does_not_remove_its_occlusion(self):
        masks = visible_source_coverage(np.array([[0, 0, 1], [0, 0, 2]], np.float32),
                                        np.array([0, 1]), self.camera(), 2, width=32, candidate_ids=[1])
        self.assertEqual(masks.shape, (1, 1024))
        self.assertFalse(masks.any())
        self.assertEqual(coverage_diagnostic(masks[None], [0])['min_coverage'], 0)

    def test_pool_ids_files_directories_and_relocated_rgb_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = root / 'rgb'
            rgb.mkdir()
            sources = [f'/old/server/frame_{i:03d}.png' for i in range(6)]
            for i in [1, 3, 5]:
                cv2.imwrite(str(rgb / f'frame_{i:03d}.png'), np.full((32, 32, 3), i, np.uint8))
            pool = resolve_reference_pool(sources, reference_ids=[5, 1, 3], reference_dir=rgb, reference_pool_size=2)
            self.assertEqual([entry['reference_id'] for entry in pool], [1, 5])
            self.assertTrue(all(Path(entry['path']).is_file() for entry in pool))
            pool = resolve_reference_pool(sources, reference_pool=rgb)
            self.assertEqual([entry['reference_id'] for entry in pool], [1, 3, 5])
            for name, text in [('ids.json', json.dumps({'reference_ids': [1, 5]})),
                               ('ids.txt', '1, 5 # source IDs\n')]:
                file = root / name
                file.write_text(text)
                pool = resolve_reference_pool(sources, reference_pool=file, reference_dir=rgb)
                self.assertEqual([entry['reference_id'] for entry in pool], [1, 5])
            for invalid in [[1, 1], [6], [-1], [], [True]]:
                with self.assertRaises(ValueError):
                    resolve_reference_pool(sources, reference_ids=invalid, reference_dir=rgb)
            with self.assertRaisesRegex(ValueError, 'fit'):
                resolve_reference_pool(sources, reference_ids=[1, 5], reference_pool_size=3, reference_dir=rgb)
            with self.assertRaisesRegex(ValueError, 'Missing candidate'):
                resolve_reference_pool(sources)
            (rgb / 'unknown.png').write_bytes(b'test')
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                resolve_reference_pool(sources, reference_pool=rgb)


if __name__ == '__main__':
    unittest.main()
