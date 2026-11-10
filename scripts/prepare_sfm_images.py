"""Create a separate, uniformly sampled low-resolution image set before SfM."""
import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--num_frames', type=int, default=400)
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--height', type=int, default=512)
    args = parser.parse_args()
    paths = sorted(p for p in args.image_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    if not 1 <= args.num_frames <= len(paths):
        parser.error('num_frames must fit the source image count')
    if min(args.width, args.height) <= 0 or args.width % 16 or args.height % 16:
        parser.error('width/height must be positive multiples of 16')
    if args.output_dir.exists():
        parser.error('Choose a new output_dir; existing images are never overwritten')
    indices = [0] if args.num_frames == 1 else [round(i * (len(paths) - 1) / (args.num_frames - 1)) for i in range(args.num_frames)]
    args.output_dir.mkdir(parents=True)
    records = []
    for index in indices:
        path = paths[index]
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot decode {path}')
        height, width = image.shape[:2]
        scale = max(args.width / width, args.height / height)
        rw, rh = round(width * scale), round(height * scale)
        image = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
        left, top = (rw - args.width) // 2, (rh - args.height) // 2
        output = args.output_dir / (path.stem + '.png')
        if output.exists():
            raise ValueError(f'Duplicate image stem: {path.stem}')
        if not cv2.imwrite(str(output), image[top:top + args.height, left:left + args.width]):
            raise OSError(f'Cannot write {output}')
        records.append(dict(source=str(path.resolve()), source_sorted_index=index, output=output.name,
                            original_size=[width, height], resized_size=[rw, rh], crop_left_top=[left, top]))
    (args.output_dir / 'selection.json').write_text(json.dumps(dict(
        image_size=[args.width, args.height], num_frames=len(records), frames=records), indent=2) + '\n')
    print(f'Prepared {len(records)} separate {args.width}x{args.height} images in {args.output_dir}')


if __name__ == '__main__':
    main()
