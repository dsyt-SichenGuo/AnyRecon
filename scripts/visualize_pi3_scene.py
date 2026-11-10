"""Regenerate the offline viewer from a scene cache without rerunning Pi3."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.reconstruction_visualization import write_scene_viewer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--max_points', type=int, default=200000)
    args = parser.parse_args()
    write_scene_viewer(args.scene, args.output or args.scene.parent / 'viewer.html', args.max_points)
