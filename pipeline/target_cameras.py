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


MAX_CHUNK_FRAMES = 1000
MAX_TRAJECTORY_SAMPLES = 20000


def target_chunk_groups(document, chunk_size):
    """Recorded boundaries take precedence; legacy camera files use fixed-size groups."""
    count = len(document['cameras'])
    if 'chunks' not in document:
        return [list(range(i, min(i + chunk_size, count))) for i in range(0, count, chunk_size)]
    groups = [chunk['target_ids'] for chunk in document['chunks']]
    if (any(not group for group in groups)
            or any(type(i) is not int for group in groups for i in group)
            or [i for group in groups for i in group] != list(range(count))):
        raise ValueError('Recorded chunks must cover all target cameras exactly once in order')
    return groups


def sample_trajectory(recording):
    """Uniform positional arc-length sampling; no timestamps or temporal weighting."""
    from scipy.spatial.transform import Rotation

    if not isinstance(recording, dict):
        raise ValueError('Recording must be a JSON object')
    count = recording['frame_count']
    if type(count) is not int or not 2 <= count <= MAX_CHUNK_FRAMES:
        raise ValueError(f'frame_count must be an integer from 2 to {MAX_CHUNK_FRAMES}')
    base = validate_camera(recording['camera'])
    samples = recording['samples']
    if not isinstance(samples, list) or not 2 <= len(samples) <= MAX_TRAJECTORY_SAMPLES:
        raise ValueError(f'Record 2..{MAX_TRAJECTORY_SAMPLES} trajectory samples')
    poses = np.asarray([validate_camera(dict(base, c2w=sample['c2w']))['c2w'] for sample in samples])
    if not np.allclose(poses[0], base['c2w'], atol=1e-8):
        raise ValueError('First sample must match the recording start camera')
    lengths = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(lengths)))
    if not np.isfinite(distances).all() or distances[-1] <= 0:
        raise ValueError('Position trajectory has zero or invalid length; move the camera before finishing')
    requested = np.linspace(0, distances[-1], count)
    # Keep both arrival/departure orientations at repeated positions. Only segments
    # with actual translation consume distance; pauses/turns in place get no frames.
    moving = np.flatnonzero(lengths > 0)
    slots = np.minimum(np.searchsorted(distances[moving + 1], requested, side='right'), len(moving) - 1)
    lower = moving[slots]
    fraction = np.clip((requested - distances[lower]) / lengths[lower], 0, 1)
    result = np.tile(np.eye(4), (count, 1, 1))
    result[:, :3, 3] = poses[lower, :3, 3] + fraction[:, None] * (poses[lower + 1, :3, 3] - poses[lower, :3, 3])
    rotations = Rotation.from_matrix(poses[:, :3, :3])
    start, end = rotations[lower], rotations[lower + 1]
    delta = (start.inv() * end).as_rotvec()
    result[:, :3, :3] = (start * Rotation.from_rotvec(fraction[:, None] * delta)).as_matrix()
    result[0], result[-1] = poses[0], poses[-1]
    cameras = [validate_camera(dict(base, c2w=pose)) for pose in result]
    return cameras, float(distances[-1]), requested.tolist(), base


class TargetCameraStore:
    """Atomic JSON is authoritative; NPZ retains flat cameras plus explicit chunk offsets."""
    def __init__(self, output_dir, scene_path, scene_id):
        self.root = Path(output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'target_cameras.json'
        self.scene_path = str(Path(scene_path).resolve())
        self.scene_id = scene_id
        self.cameras = []
        self.chunks = []
        if self.path.exists():
            document = json.loads(self.path.read_text())
            if document['scene_id'] != scene_id:
                raise ValueError('Output directory already belongs to a different Pi3 reconstruction')
            self.cameras = [validate_camera(c) for c in document['cameras']]
            groups = target_chunk_groups(document, max(1, len(self.cameras)))
            self.chunks = [dict(document.get('chunks', [{}] * len(groups))[i],
                                chunk_id=i, target_ids=group, frame_count=len(group))
                           for i, group in enumerate(groups)]
            self._write_npz()

    def document(self):
        return dict(schema_version=2, scene_path=self.scene_path, scene_id=self.scene_id,
                    camera_coordinates='OpenCV: +X right, +Y down, +Z forward',
                    world_coordinates='Unmodified source Pi3 world; arbitrary scale',
                    image_size_order='width,height', chunks=self.chunks,
                    cameras=[dict(frame_id=i, **c) for i, c in enumerate(self.cameras)])

    def _write_npz(self):
        with (self.root / 'target_cameras.npz.tmp').open('wb') as stream:
            np.savez_compressed(stream, schema_version=np.int32(2),
                                scene_path=self.scene_path, scene_id=self.scene_id,
                                c2w=np.asarray([c['c2w'] for c in self.cameras], dtype=np.float64).reshape(-1, 4, 4),
                                w2c=np.asarray([c['w2c'] for c in self.cameras], dtype=np.float64).reshape(-1, 4, 4),
                                K=np.asarray([c['K'] for c in self.cameras], dtype=np.float64).reshape(-1, 3, 3),
                                image_size=np.asarray([c['image_size'] for c in self.cameras], dtype=np.int32).reshape(-1, 2),
                                chunk_offsets=np.asarray([0] + [c['target_ids'][-1] + 1 for c in self.chunks], dtype=np.int64),
                                chunk_ids=np.asarray([i for i, c in enumerate(self.chunks) for _ in c['target_ids']], dtype=np.int32))
        (self.root / 'target_cameras.npz.tmp').replace(self.root / 'target_cameras.npz')

    def _persist(self, cameras, chunks):
        previous = self.cameras, self.chunks
        self.cameras, self.chunks = cameras, chunks
        try:
            temporary = self.path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(self.document(), indent=2) + '\n')
            temporary.replace(self.path)
        except Exception:
            self.cameras, self.chunks = previous
            raise
        self._write_npz()

    def append(self, camera):
        """Compatibility endpoint for a single manually saved camera."""
        validated = validate_camera(camera)
        start = len(self.cameras)
        chunk = dict(chunk_id=len(self.chunks), target_ids=[start], frame_count=1,
                     kind='single', restore_camera=self.cameras[-1] if self.cameras else validated)
        self._persist(self.cameras + [validated], self.chunks + [chunk])
        return start

    def append_chunk(self, recording):
        if not isinstance(recording, dict):
            raise ValueError('Recording must be a JSON object')
        recording_id = recording.get('recording_id')
        if recording_id is not None and (not isinstance(recording_id, str) or not 1 <= len(recording_id) <= 128):
            raise ValueError('Invalid recording_id')
        fingerprint = hashlib.sha256(json.dumps(recording, sort_keys=True).encode()).hexdigest()
        for existing in self.chunks:
            if recording_id is not None and existing.get('recording_id') == recording_id:
                if existing['recording_sha256'] != fingerprint:
                    raise ValueError('recording_id already belongs to a different trajectory')
                self._write_npz()
                return existing
        cameras, path_length, distances, restore = sample_trajectory(recording)
        if self.cameras and cameras[0]['image_size'] != self.cameras[0]['image_size']:
            raise ValueError('All recorded chunks must share one output image_size')
        start = len(self.cameras)
        chunk = dict(chunk_id=len(self.chunks), target_ids=list(range(start, start + len(cameras))),
                     frame_count=len(cameras), path_length=path_length, sample_distances=distances,
                     sampling='position_arc_length',
                     kind='recorded', restore_camera=restore)
        if recording_id is not None:
            chunk.update(recording_id=recording_id, recording_sha256=fingerprint)
        self._persist(self.cameras + cameras, self.chunks + [chunk])
        return chunk

    def undo(self):
        if not self.chunks:
            raise ValueError('No saved chunk to undo')
        chunk = self.chunks[-1]
        start = chunk['target_ids'][0]
        restore = chunk.get('restore_camera', self.cameras[max(0, start - 1)])
        self._persist(self.cameras[:start], self.chunks[:-1])
        return validate_camera(restore)
