"""Check the actual AnyRecon inference imports, tokenizer and CUDA runtime."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from diffsynth import ModelManager
from pipeline.anyrecon_pipeline import AnyReconPipeline
from Pi3.pi3.models.pi3 import Pi3
import geometry_utils
from transformers import AutoTokenizer

print('Python:', sys.version)
print('Torch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
print('NumPy:', np.__version__, 'OpenCV:', cv2.__version__)
print('Inference imports: OK')
tokenizer = AutoTokenizer.from_pretrained(
    ROOT / 'checkpoints/google/umt5-xxl', local_files_only=True
)
assert tokenizer('a room', return_tensors='pt')['input_ids'].numel() > 0
print('Local tokenizer: OK')
ffmpeg = shutil.which('ffmpeg')
assert ffmpeg, 'ffmpeg missing from PATH'
encoders = subprocess.check_output([ffmpeg, '-hide_banner', '-encoders'], stderr=subprocess.STDOUT).decode()
assert 'libx264 ' in encoders, 'FFmpeg must include libx264'
print('FFmpeg libx264: OK')
assert torch.cuda.is_available(), 'CUDA device unavailable'
a = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
b = a @ a.T
torch.cuda.synchronize()
assert torch.isfinite(b).all()
print('CUDA bfloat16 matmul: OK, device:', torch.cuda.get_device_name())
manifest = json.loads((ROOT / 'setup_logs/model_manifest.json').read_text())
missing = [i['path'] for i in manifest if not (ROOT / i['path']).is_file()
           or (ROOT / i['path']).stat().st_size != i['size']]
if missing:
    raise RuntimeError(f'Model files not complete: {missing}')
print(f'Model/tokenizer files: {len(manifest)} complete (checksums in download logs)')
