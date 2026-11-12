import json
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch  # Import before patch.dict(sys.modules), so repeated entry runs keep it cached.
from PIL import Image

from pipeline.reference_selection import visible_source_coverage, choose_chunk_references
from pipeline.target_cameras import scene_identity


class SelectedTargetPreparationTests(unittest.TestCase):
    def test_retrieval_discards_occluded_source(self):
        camera = dict(w2c=np.eye(4).tolist(), K=[[16, 0, 16], [0, 16, 16], [0, 0, 1]], image_size=[32, 32])
        points = np.array([[0, 0, 1], [0, 0, 2], [.25, 0, 1]], dtype=np.float32)
        masks = visible_source_coverage(points, np.array([0, 1, 2]), camera, 3, width=32)
        np.testing.assert_array_equal(masks.sum(axis=1), [1, 0, 1])

    def test_selection_seeds_each_target_then_adds_complementary_coverage(self):
        masks = np.zeros((4, 8, 16), dtype=bool)
        for target in range(4):
            masks[target, target, :4] = True
            masks[target, 4, 4:6] = True
            masks[target, 5, 6:8] = True
            masks[target, 6, :3] = True  # redundant image
        selected, report = choose_chunk_references(masks, 6)
        self.assertEqual(selected, [0, 1, 2, 3, 4, 5])
        self.assertEqual(report['selected_coverage_fraction_per_target'], [1., 1., 1., 1.])
        with self.assertRaisesRegex(ValueError, 'no sampled visible'):
            choose_chunk_references(np.zeros((1, 2, 16), dtype=bool), 1)

    def test_thirty_targets_can_share_ten_references(self):
        masks = np.zeros((30, 40, 16), dtype=bool)
        for target in range(30):
            masks[target, target // 3, :8] = True
            masks[target, 10 + target, 8:] = True
        selected, report = choose_chunk_references(masks, 10)
        self.assertEqual(len(set(selected)), 10)
        self.assertTrue(all(fraction > 0 for fraction in report['selected_coverage_fraction_per_target']))

    def test_cli_plan_resume_render_encode_and_mapping(self):
        root_repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            poses = np.repeat(np.eye(4)[None], 8, axis=0)
            K = np.repeat(np.array([[[32., 0, 32], [0, 32, 24], [0, 0, 1]]]), 8, axis=0)
            paths = []
            for i in range(8):
                path = root / f'ref_{i:03d}.png'
                cv2.imwrite(str(path), np.full((48, 64, 3), 50 + i * 20, dtype=np.uint8))
                paths.append(str(path))
            points = np.column_stack([np.linspace(-.8, .8, 16), np.zeros(16), np.full(16, 2)]).astype(np.float32)
            np.savez(root / 'scene.npz', points=points, colors=np.full((16, 3), 200, dtype=np.uint8),
                     source_indices=np.repeat(np.arange(8), 2), input_c2w=poses, intrinsics_reconstruction=K,
                     image_size_reconstruction=[64, 48], num_reference_frames=8, frame_sources=paths)
            identity = scene_identity(poses, K, [64, 48])
            document = dict(scene_id=identity, cameras=[dict(c2w=p.tolist(), K=k.tolist(), image_size=[64, 48]) for p, k in zip(poses, K)])
            (root / 'cameras.json').write_text(json.dumps(document))
            command = [sys.executable, str(root_repo / 'scripts/prepare_selected_targets.py'),
                       '--scene', str(root / 'scene.npz'), '--cameras', str(root / 'cameras.json'),
                       '--output_dir', str(root / 'prepared'), '--num_reference', '6',
                       '--retrieval_points', '16', '--retrieval_width', '32']
            environment = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
            subprocess.run(command + ['--plan_only'], env=environment, check=True, capture_output=True)
            manifest = json.loads((root / 'prepared/manifest.json').read_text())
            self.assertEqual([c['target_ids'] for c in manifest['chunks']], [[0, 1, 2, 3], [4, 5, 6, 7]])
            self.assertEqual(list((root / 'prepared/condition').glob('*.mp4')), [])
            # Simulate an interrupted reference plan: reuse the persisted first
            # chunk, compute only the missing second chunk, and retain its timing.
            partial = dict(manifest, chunks=manifest['chunks'][:1])
            (root / 'prepared/manifest.json').write_text(json.dumps(partial))
            resumed_plan = subprocess.run(command + ['--plan_only'], env=environment,
                                          check=True, capture_output=True, text=True)
            self.assertNotIn('Retrieving chunk 1/2', resumed_plan.stdout)
            self.assertIn('Retrieving chunk 2/2', resumed_plan.stdout)
            repaired = json.loads((root / 'prepared/manifest.json').read_text())
            self.assertEqual(repaired['chunks'][0], manifest['chunks'][0])
            self.assertEqual(len(repaired['chunks']), 2)
            subprocess.run(command, env=environment, check=True, capture_output=True)
            for chunk in manifest['chunks']:
                for folder in ['condition', 'mask']:
                    cap = cv2.VideoCapture(str(root / 'prepared' / folder / chunk['video']))
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 10)
                    self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 64)
                    cap.release()
                info = (root / 'prepared/condition' / (Path(chunk['video']).stem + '_info.txt')).read_text()
                self.assertIn('Condition Frame Count: 6', info)
                self.assertIn('Render Frame Count: 4', info)
            resumed = subprocess.run(command, env=environment, check=True, capture_output=True, text=True)
            self.assertIn('already prepared', resumed.stdout)
            # Exercise the actual inference entry point with only model calls replaced.
            # Verify independent chunks stay at 10 frames and prefix removal is exact.
            pipe = MagicMock()
            pipe.side_effect = lambda **kw: [Image.new('RGB', (64, 48), (i, 0, 0)) for i in range(kw['num_frames'])]
            fake_diff = SimpleNamespace(ModelManager=MagicMock(), VideoData=object,
                                        save_video=lambda frames, path, **kw: Path(path).write_bytes(b'test'))
            fake_pipeline = SimpleNamespace(AnyReconPipeline=SimpleNamespace(from_model_manager=lambda *a, **kw: pipe))
            argv = ['run_AnyRecon.py', '--root_dir', str(root / 'prepared'), '--output_dir', str(root / 'generated'),
                    '--lora_path', 'unused.ckpt', '--width', '64', '--height', '48',
                    '--independent_chunks', '--export_target_frames']
            with patch.dict(sys.modules, {'diffsynth': fake_diff, 'pipeline.anyrecon_pipeline': fake_pipeline}), \
                    patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                runpy.run_path(str(root_repo / 'run_AnyRecon.py'), run_name='__main__')
            self.assertEqual([call.kwargs['num_frames'] for call in pipe.call_args_list], [10, 10])
            mapping = json.loads((root / 'generated/target_mapping.json').read_text())
            self.assertEqual([row['target_id'] for row in mapping], list(range(8)))
            for target_id in range(8):
                with Image.open(root / 'generated/target_frames' / f'target_{target_id:06d}.png') as image:
                    self.assertEqual(image.getpixel((0, 0)), (6 + target_id % 4, 0, 0))
            manifest['independent_chunks'] = False
            (root / 'prepared/manifest.json').write_text(json.dumps(manifest))
            pipe.reset_mock()
            context_argv = [arg for arg in argv if arg != '--independent_chunks'] + ['--context_chunks']
            context_argv[context_argv.index('--output_dir') + 1] = str(root / 'generated_context')
            with patch.dict(sys.modules, {'diffsynth': fake_diff, 'pipeline.anyrecon_pipeline': fake_pipeline}), \
                    patch.object(sys, 'argv', context_argv), redirect_stdout(io.StringIO()):
                runpy.run_path(str(root_repo / 'run_AnyRecon.py'), run_name='__main__')
            self.assertEqual([call.kwargs['num_frames'] for call in pipe.call_args_list], [10, 12])
            second = pipe.call_args_list[1].kwargs
            self.assertEqual(second['input_video'][6].getpixel((0, 0)), (8, 0, 0))
            self.assertEqual(second['input_video'][7].getpixel((0, 0)), (9, 0, 0))
            self.assertTrue((second['mask_frames'][6].numpy() == 255).all())
            context_mapping = json.loads((root / 'generated_context/target_mapping.json').read_text())
            self.assertEqual([r['chunk_frame_index'] for r in context_mapping], [6, 7, 8, 9, 8, 9, 10, 11])
            for target_id in range(4, 8):
                with Image.open(root / 'generated_context/target_frames' / f'target_{target_id:06d}.png') as image:
                    self.assertEqual(image.getpixel((0, 0)), (8 + target_id % 4, 0, 0))
            bad = subprocess.run(command + ['--num_reference', '5'], env=environment, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            # Recorded chunk boundaries must survive even when --chunk_size differs.
            document['chunks'] = [dict(target_ids=[0, 1]), dict(target_ids=list(range(2, 8)))]
            (root / 'cameras.json').write_text(json.dumps(document))
            recorded_command = command.copy()
            recorded_command[recorded_command.index('--output_dir') + 1] = str(root / 'recorded_prepared')
            subprocess.run(recorded_command + ['--plan_only'], env=environment, check=True, capture_output=True)
            recorded = json.loads((root / 'recorded_prepared/manifest.json').read_text())
            self.assertEqual([c['target_ids'] for c in recorded['chunks']], [[0, 1], list(range(2, 8))])
            # Both methods package identical targets once, with a restricted original-ID pool.
            comparison_command = recorded_command.copy()
            comparison_command[comparison_command.index('--output_dir') + 1] = str(root / 'comparison')
            comparison_command[comparison_command.index('--num_reference') + 1] = '2'
            comparison_command += ['--reference_method', 'compare', '--reference_ids', '1', '3', '5', '7',
                                   '--reference_pool_size', '3']
            subprocess.run(comparison_command, env=environment, check=True, capture_output=True)
            self.assertEqual(len(list((root / 'comparison/renders/rgb').glob('*.png'))), 8)
            self.assertTrue((root / 'comparison/selection_comparison.html').is_file())
            report = json.loads((root / 'comparison/selection_comparison.json').read_text())
            self.assertEqual(set(report['chunks'][0]['methods']), {'original', 'improved'})
            pool = json.loads((root / 'comparison/reference_pool.json').read_text())
            self.assertEqual([entry['reference_id'] for entry in pool['candidates']], [1, 5, 7])
            for method in ['original', 'improved']:
                prepared = root / 'comparison' / method
                manifest = json.loads((prepared / 'manifest.json').read_text())
                self.assertEqual([c['target_ids'] for c in manifest['chunks']], [[0, 1], list(range(2, 8))])
                for chunk in manifest['chunks']:
                    self.assertEqual(len(chunk['reference_ids']), 2)
                    self.assertTrue(set(chunk['reference_ids']) <= {1, 5, 7})
                    metadata = json.loads((prepared / chunk['camera_metadata']).read_text())
                    self.assertEqual([f['role'] for f in metadata['frames']],
                                     ['reference']*2 + ['target']*len(chunk['target_ids']))
                    for frame in metadata['frames']:
                        self.assertTrue((prepared / frame['image']).is_file())
                        self.assertTrue((prepared / frame['mask']).is_file())
                        self.assertEqual(frame['image_size'], [64, 48])
                        np.testing.assert_allclose(np.asarray(frame['w2c']) @ frame['c2w'], np.eye(4))
                    for frame in chunk['reference_frames']:
                        image = cv2.imread(str(prepared / frame['image']))
                        self.assertEqual(int(image[0, 0, 0]), 50 + frame['reference_id']*20)
                        self.assertTrue((cv2.imread(str(prepared / frame['mask'])) == 255).all())
                    for folder in ['condition', 'mask']:
                        cap = cv2.VideoCapture(str(prepared / folder / chunk['video']))
                        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 2 + len(chunk['target_ids']))
                        cap.release()
                self.assertTrue((prepared / 'generation_command.txt').is_file())
                # The unchanged generation entry point accepts either method's artifacts.
                pipe.reset_mock()
                compare_argv = argv.copy()
                compare_argv[compare_argv.index('--root_dir') + 1] = str(prepared)
                compare_argv[compare_argv.index('--output_dir') + 1] = str(root / f'{method}_generated')
                with patch.dict(sys.modules, {'diffsynth': fake_diff, 'pipeline.anyrecon_pipeline': fake_pipeline}), \
                        patch.object(sys, 'argv', compare_argv), redirect_stdout(io.StringIO()):
                    runpy.run_path(str(root_repo / 'run_AnyRecon.py'), run_name='__main__')
                self.assertEqual([call.kwargs['num_frames'] for call in pipe.call_args_list], [4, 8])
                self.assertEqual(len(list((root / f'{method}_generated/target_frames').glob('*.png'))), 8)
            resumed = subprocess.run(comparison_command, env=environment, check=True, capture_output=True, text=True)
            self.assertIn('already prepared', resumed.stdout)
            # Missing exported references invalidate completion and are repaired from the source.
            victim = root / 'comparison/improved' / manifest['chunks'][0]['reference_frames'][0]['image']
            victim.unlink()
            subprocess.run(comparison_command, env=environment, check=True, capture_output=True)
            self.assertTrue(victim.is_file())
            # Exercise CUDA cache switching/repacking on CPU-only CI using the exact
            # tensor algorithm on CPU. Hardware CUDA parity is a separate optional test.
            from pipeline.point_cloud_renderer import TensorSplatRenderer
            def tensor_backend(points, colors, device, batch_size, max_splats):
                self.assertEqual(device, 'cuda')
                tensor = TensorSplatRenderer(points, colors, 'cpu', batch_size, max_splats)
                return SimpleNamespace(backend='cuda_alpha_over_v1', render=tensor.render, close=lambda: None)
            cuda_argv = comparison_command[1:] + ['--render_device', 'cuda', '--render_batch_size', '5']
            with patch('torch.cuda.is_available', return_value=True), \
                    patch('pipeline.point_cloud_renderer.PointCloudRenderer', side_effect=tensor_backend) as factory, \
                    patch.object(sys, 'argv', cuda_argv), redirect_stdout(io.StringIO()):
                runpy.run_path(str(root_repo / 'scripts/prepare_selected_targets.py'), run_name='__main__')
            self.assertEqual(factory.call_count, 1)  # Reuse one uploaded cloud for all targets.
            for target_id in range(8):
                meta = json.loads((root / f'comparison/renders/metadata/target_{target_id:06d}.json').read_text())
                self.assertEqual(meta['backend'], 'cuda_alpha_over_v1')
            for method in ['original', 'improved']:
                for i in range(2):
                    status = json.loads((root / f'comparison/{method}/status/chunk_{i:03d}.json').read_text())
                    self.assertEqual(status['render_device'], 'cuda')
            with patch('torch.cuda.is_available', return_value=True), \
                    patch('pipeline.point_cloud_renderer.PointCloudRenderer') as factory, \
                    patch.object(sys, 'argv', cuda_argv), redirect_stdout(io.StringIO()) as output:
                runpy.run_path(str(root_repo / 'scripts/prepare_selected_targets.py'), run_name='__main__')
            factory.assert_not_called()
            self.assertIn('already prepared', output.getvalue())


if __name__ == '__main__':
    unittest.main()
