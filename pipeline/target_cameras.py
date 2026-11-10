"""Validated target-camera interchange in the unchanged Pi3 world frame."""
import hashlib
import json
from pathlib import Path

import numpy as np


def scene_identity(poses, intrinsics, image_size):
    digest = hashlib.sha256()
    for value in (poses, intrinsics, image_size):
        array = np.ascontiguousarray(value, dtype='<f8')
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def validate_camera(camera):
    c2w = np.asarray(camera['c2w'], dtype=np.float64)
    K = np.asarray(camera['K'], dtype=np.float64)
    size = np.asarray(camera['image_size'], dtype=np.float64)
    if c2w.shape != (4, 4) or K.shape != (3, 3) or size.shape != (2,):
        raise ValueError('Expected c2w[4,4], K[3,3], image_size[width,height]')
    if not all(np.isfinite(a).all() for a in (c2w, K, size)):
        raise ValueError('Camera contains non-finite values')
    if not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError('Invalid homogeneous pose row')
    rotation = c2w[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError('Camera rotation must be orthonormal and right-handed')
    if not np.allclose(K[2], [0, 0, 1]) or K[0, 0] <= 0 or K[1, 1] <= 0 or abs(K[0, 1]) > 1e-8 or abs(K[1, 0]) > 1e-8:
        raise ValueError('Expected positive focal lengths and zero-skew pinhole K')
    if np.any(size < 16) or np.any(size > 16384) or np.any(size != np.floor(size)) or np.any(size % 16):
        raise ValueError('AnyRecon output width/height must be multiples of 16 (16..16384)')
    return dict(c2w=c2w.tolist(), w2c=np.linalg.inv(c2w).tolist(),
                K=K.tolist(), image_size=size.astype(int).tolist())


class TargetCameraStore:
    """One atomic JSON manifest is authoritative; NPZ is rebuilt on startup/save."""
    def __init__(self, output_dir, scene_path, scene_id):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'target_cameras.json'
        self.scene_path = str(Path(scene_path).resolve())
        self.scene_id = scene_id
        self.cameras = []
        if self.path.exists():
            document = json.loads(self.path.read_text())
            if document['scene_id'] != scene_id:
                raise ValueError('Output directory already belongs to a different Pi3 reconstruction')
            self.cameras = [validate_camera(c) for c in document['cameras']]
            self._write_npz()

    def document(self):
        return dict(schema_version=1, scene_path=self.scene_path, scene_id=self.scene_id,
                    camera_coordinates='OpenCV: +X right, +Y down, +Z forward',
                    world_coordinates='Unmodified source Pi3 world; arbitrary scale',
                    image_size_order='width,height',
                    cameras=[dict(frame_id=i, **c) for i, c in enumerate(self.cameras)])

    def _write_npz(self):
        # Numeric arrays can be passed directly to render_point_cloud_to_memory.
        with (self.root / 'target_cameras.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, schema_version=np.int32(1),
                                scene_path=self.scene_path, scene_id=self.scene_id,
                                c2w=np.asarray([c['c2w'] for c in self.cameras], dtype=np.float64).reshape(-1, 4, 4),
                                w2c=np.asarray([c['w2c'] for c in self.cameras], dtype=np.float64).reshape(-1, 4, 4),
                                K=np.asarray([c['K'] for c in self.cameras], dtype=np.float64).reshape(-1, 3, 3),
                                image_size=np.asarray([c['image_size'] for c in self.cameras], dtype=np.int32).reshape(-1, 2))
        (self.root / 'target_cameras.npz.tmp').replace(self.root / 'target_cameras.npz')

    def append(self, camera):
        validated = validate_camera(camera)
        self.cameras.append(validated)
        try:
            temporary = self.path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(self.document(), indent=2) + '\n')
            temporary.replace(self.path)
        except Exception:
            self.cameras.pop()
            raise
        self._write_npz()
        return len(self.cameras) - 1
