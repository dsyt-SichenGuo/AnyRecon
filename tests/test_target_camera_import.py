import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from pipeline.target_cameras import TargetCameraStore
from scripts.select_target_cameras import main


class TargetCameraImportTests(unittest.TestCase):
    def payload(self, x, recording_id):
        camera = dict(c2w=np.eye(4).tolist(), K=[[80, 0, 32], [0, 90, 24], [0, 0, 1]], image_size=[64, 48])
        end = copy.deepcopy(camera)
        end['c2w'][0][3] = x
        return dict(kind='advance_panorama', camera=end, start_camera=camera,
                    advance_frames=5, panorama_frames=25, world_up=[0, -1, 0], recording_id=recording_id)

    def test_import_preserves_metadata_append_undo_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TargetCameraStore(root / 'source', '/tmp/scene.npz', 'same-world')
            source.append_chunk(self.payload(3, 'first'))
            source.append_chunk(self.payload(6, 'second'))
            before = source.path.read_bytes()
            destination = TargetCameraStore(root / 'continued', '/tmp/scene.npz', 'same-world')
            doc = json.loads(before)
            destination.import_document(doc)
            self.assertEqual(destination.chunks, source.chunks)
            self.assertEqual(destination.cameras, source.cameras)
            destination.import_document(doc)  # Idempotent import retry.
            chunk = destination.append_chunk(self.payload(9, 'third'))
            self.assertEqual(chunk['chunk_id'], 2)
            self.assertEqual(chunk['target_ids'], list(range(60, 90)))
            self.assertEqual(destination.cameras[60], source.cameras[-1])
            self.assertEqual(destination.undo(), source.cameras[-1])
            restored = TargetCameraStore(root / 'continued', '/tmp/scene.npz', 'same-world')
            self.assertEqual(restored.chunks, source.chunks)
            with np.load(root / 'continued/target_cameras.npz', allow_pickle=False) as data:
                np.testing.assert_array_equal(data['chunk_offsets'], [0, 30, 60])
            self.assertEqual(source.path.read_bytes(), before)

    def test_reject_wrong_scene_invalid_boundaries_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TargetCameraStore(root / 'source', '/tmp/scene.npz', 'same-world')
            source.append_chunk(self.payload(3, 'first'))
            good = source.document()
            target = TargetCameraStore(root / 'target', '/tmp/scene.npz', 'same-world')
            bad = copy.deepcopy(good)
            bad['chunks'][0]['target_ids'] = [0, 1]
            malformed = copy.deepcopy(good)
            malformed['chunks'][0]['advance_frames'] = 4
            for doc in [dict(good, scene_id='wrong'), dict(good, cameras=[]), bad, malformed]:
                with self.assertRaises(ValueError):
                    target.import_document(doc)
                self.assertEqual(target.cameras, [])
                self.assertFalse(target.path.exists())
            target.append_chunk(self.payload(8, 'other'))
            before = target.path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'already has another'):
                target.import_document(good)
            self.assertEqual(target.path.read_bytes(), before)

    def test_legacy_cameras_without_chunks_can_be_extended(self):
        with tempfile.TemporaryDirectory() as directory:
            target = TargetCameraStore(directory, '/tmp/scene.npz', 'same-world')
            camera = self.payload(1, 'first')['camera']
            target.import_document(dict(scene_id='same-world', cameras=[camera]))
            chunk = target.append_chunk(self.payload(4, 'next'))
            self.assertEqual(chunk['target_ids'], list(range(1, 31)))
            self.assertEqual(len(target.chunks), 2)

    def test_cli_accepts_directory_or_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = TargetCameraStore(root / 'source', '/tmp/scene.npz', 'same-world')
            source.append_chunk(self.payload(3, 'first'))
            for i, path in enumerate([source.root, source.path]):
                server = MagicMock()
                server.serve_forever.side_effect = KeyboardInterrupt
                output = root / f'continued_{i}'
                with patch('sys.argv', ['select_target_cameras.py', '--scene', '/tmp/scene.npz',
                                       '--output_dir', str(output), '--load_targets', str(path)]), \
                        patch('scripts.select_target_cameras.load_viewer_scene',
                              return_value=({'shown_points': 0, 'total_points': 0}, b'', 'same-world')), \
                        patch('scripts.select_target_cameras.HTTPServer', return_value=server), \
                        redirect_stdout(io.StringIO()):
                    main()
                loaded = json.loads((output / 'target_cameras.json').read_text())
                self.assertEqual(len(loaded['cameras']), 30)
                self.assertEqual(loaded['chunks'], source.chunks)


if __name__ == '__main__':
    unittest.main()
