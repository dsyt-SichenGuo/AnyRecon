"""Download the pinned official inference weights and verify their SHA256."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def download(item):
    target = ROOT / item['path']
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    if not target.exists():
        url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['file']}"
        print(f"Downloading {item['path']}", flush=True)
        if os.environ.get('ANYRECON_DOWNLOADER') == 'aria2':
            executable = shutil.which('aria2c')
            if not executable:
                raise RuntimeError('aria2c is not on PATH')
            subprocess.run([executable, '--continue=true', '--max-connection-per-server=8',
                            '--split=8', '--min-split-size=16M', '--max-tries=8',
                            '--retry-wait=5', '--file-allocation=none',
                            '--auto-file-renaming=false', '--allow-overwrite=true',
                            '--console-log-level=warn', '--summary-interval=0',
                            '--download-result=hide', '--dir', str(target.parent),
                            '--out', partial.name, url], check=True)
        else:
            subprocess.run(['curl', '-fL', '--silent', '--show-error', '--retry', '8',
                            '--retry-delay', '5', '--connect-timeout', '30',
                            '-C', '-', '-o', str(partial), url], check=True)
        candidate = partial
    else:
        candidate = target
    if candidate.stat().st_size != item['size']:
        raise RuntimeError(f"Size mismatch: {candidate}")
    digest = hashlib.sha256() if item.get('sha256') else hashlib.sha1()
    if not item.get('sha256'):
        digest.update(f"blob {item['size']}\0".encode())
    with candidate.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != (item.get('sha256') or item['blob_id']):
        raise RuntimeError(f"Checksum mismatch: {candidate}")
    if candidate != target:
        candidate.replace(target)
    print(f"Verified {item['path']}", flush=True)
    return item['path']


if __name__ == '__main__':
    manifest = json.loads((ROOT / 'setup_logs/model_manifest.json').read_text())
    workers = 4 if os.environ.get('ANYRECON_DOWNLOADER') == 'aria2' else 12
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(download, manifest))
    print(f'All {len(manifest)} model/tokenizer files downloaded and verified.', flush=True)
