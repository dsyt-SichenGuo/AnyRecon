import torch
from diffsynth import ModelManager, save_video, VideoData
from pipeline.anyrecon_pipeline import AnyReconPipeline
from PIL import Image
import cv2
import os
import argparse
import numpy as np
import re
import json
from pathlib import Path

parser = argparse.ArgumentParser(description="Wan Video Pipeline with custom paths")
parser.add_argument("--root_dir", type=str, required=True, help="Root directory containing condition and mask folders")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save results")
parser.add_argument("--is_block", action="store_true", help="Whether to use block attention")
parser.add_argument("--lora_path", type=str, required=True, help="Path to the LoRA checkpoint")
parser.add_argument("--width", type=int, default=896)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--num_inference_steps", type=int, default=4)
parser.add_argument("--seed", type=int, default=1)
context_group = parser.add_mutually_exclusive_group()
context_group.add_argument("--independent_chunks", action="store_true", help="Disable previous generated frame carry between chunks")
context_group.add_argument("--context_chunks", action="store_true", help="Carry the previous chunk's last two generated frames (default behavior)")
parser.add_argument("--export_target_frames", action="store_true", help="Save target-only PNGs using prepared manifest.json")
parser.add_argument("--chunk_ids", type=int, nargs='+', help="Process only these chunk IDs (requires prepared manifest)")
args = parser.parse_args()
if args.width <= 0 or args.height <= 0 or args.width % 16 or args.height % 16:
    parser.error('width and height must be positive multiples of 16')
if args.num_inference_steps <= 0:
    parser.error('num_inference_steps must be positive')

prepared_manifest = None
if args.export_target_frames or args.chunk_ids is not None:
    manifest_path = Path(args.root_dir) / 'manifest.json'
    if not manifest_path.is_file():
        parser.error('This mode requires manifest.json from scripts/prepare_selected_targets.py')
    prepared_manifest = json.loads(manifest_path.read_text())
    if prepared_manifest['image_size'] != [args.width, args.height]:
        parser.error('Requested generation size must match prepared camera image_size')
    if args.is_block:
        parser.error('Prepared target export requires full attention; omit --is_block to avoid frame truncation')
    if bool(prepared_manifest['independent_chunks']) != args.independent_chunks:
        parser.error('Chunk context mode differs from preparation; use --independent_chunks or --context_chunks accordingly')
    if args.chunk_ids is not None and (len(set(args.chunk_ids)) != len(args.chunk_ids) or
            any(i < 0 or i >= len(prepared_manifest['chunks']) for i in args.chunk_ids)):
        parser.error('Invalid or duplicate chunk_ids')
    selected = [c for c in prepared_manifest['chunks'] if args.chunk_ids is None or c['chunk_id'] in args.chunk_ids]
    if not args.independent_chunks and [c['chunk_id'] for c in selected] != list(range(len(selected))):
        parser.error('Context mode must process consecutive chunks starting at zero')
    for chunk in selected:
        for folder in ['condition', 'mask']:
            if not (Path(args.root_dir) / folder / chunk['video']).is_file():
                parser.error(f'Chunk {chunk["chunk_id"]} is not prepared: missing {folder}/{chunk["video"]}')
        if not (Path(args.root_dir) / 'status' / f'chunk_{chunk["chunk_id"]:03d}.json').is_file():
            parser.error(f'Chunk {chunk["chunk_id"]} preparation is incomplete')
    if args.export_target_frames and (Path(args.output_dir) / 'target_frames').exists():
        parser.error('Target export already exists; choose a new output_dir')

model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")

model_manager.load_models(
    ["./checkpoints/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"],
    torch_dtype=torch.float32,
)

model_manager.load_models(
    [
        [
            "./checkpoints/diffusion_pytorch_model-00001-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00002-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00003-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00004-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00005-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00006-of-00007.safetensors",
            "./checkpoints/diffusion_pytorch_model-00007-of-00007.safetensors",
        ],
        "./checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
        "./checkpoints/Wan2.1_VAE.pth",
    ],
    torch_dtype=torch.bfloat16,
)

model_manager.load_lora(args.lora_path, lora_alpha=1.0)
pipe = AnyReconPipeline.from_model_manager(model_manager, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

condition_dir = os.path.join(args.root_dir, "condition")
mask_dir = os.path.join(args.root_dir, "mask")

os.makedirs(args.output_dir, exist_ok=True)

video_files = sorted(
    [f for f in os.listdir(condition_dir) if f.endswith('.mp4')],
    key=lambda name: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)],
)
if prepared_manifest is not None:
    selected_names = {c['video'] for c in selected}
    video_files = [name for name in video_files if name in selected_names]
target_mapping = []
if args.export_target_frames:
    (Path(args.output_dir) / 'target_frames').mkdir()
    (Path(args.output_dir) / 'target_cameras.json').write_text(json.dumps(dict(
        scene_id=prepared_manifest['scene_id'], cameras=prepared_manifest['target_cameras']), indent=2) + '\n')

previous_frames = None

for video_file in video_files:
    if args.independent_chunks:
        previous_frames = None
    video_path = os.path.join(condition_dir, video_file)
    mask_pth = os.path.join(mask_dir, video_file)
    
    print(f"Processing {video_file}...")

    cap = cv2.VideoCapture(video_path)
    video = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        video.append(pil_frame)
    cap.release()
    
    info_txt_path = os.path.join(condition_dir, video_file.rsplit('.mp4', 1)[0] + '_info.txt')

    insert_index = None
    if os.path.exists(info_txt_path):
        try:
            with open(info_txt_path, 'r') as f:
                for line in f:
                    if "Condition Frame Count:" in line:
                        insert_index = int(line.split(":")[1].strip())
                        print(f"Read insert_index {insert_index} from {info_txt_path}")
                        break
        except Exception as e:
            print(f"Error reading info txt {info_txt_path}: {e}")

    if insert_index is None or not 0 < insert_index <= len(video):
        raise ValueError(f'Missing or invalid Condition Frame Count in {info_txt_path}')

    if previous_frames is not None:
        if len(video) >= insert_index:
            video.insert(insert_index, previous_frames[0])
            video.insert(insert_index + 1, previous_frames[1])
            print(f"Inserted previous last frames at index {insert_index} and {insert_index + 1} of {video_file} (frames shifted)")
        else:
            print(f"Warning: Video {video_file} is too short to insert frame at index {insert_index}")

    image = video[0]

    cap = cv2.VideoCapture(mask_pth)
    mask_video = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        pil_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        mask_video.append(pil_frame)
    cap.release()
    if len(mask_video) != len(video) - (2 if previous_frames is not None else 0):
        raise ValueError(f'Condition/mask frame counts do not match for {video_file}')
    
    if previous_frames is not None:
         if len(mask_video) >= insert_index:
            ref_mask = mask_video[0]
            white_mask = torch.ones_like(ref_mask) * 255
            mask_video.insert(insert_index, white_mask)
            mask_video.insert(insert_index + 1, white_mask)
            print(f"Inserted white masks at index {insert_index} and {insert_index + 1} of {video_file} (frames shifted)")

    if args.is_block:
        num_frames = len(video) // 8 * 8 
    else:
        num_frames = len(video)

    video = video[:num_frames]
    mask_video = mask_video[:num_frames]
    video_output = pipe(
        prompt=" ",
        negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        input_video=video,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed, num_frames=num_frames, tiled=True,
        height=args.height, width=args.width,
        mask_indices=[0], 
        mask_frames=mask_video,
        is_block=args.is_block
    )
    
    root_dir_name = os.path.basename(os.path.normpath(args.root_dir))
    output_save_path = os.path.join(args.output_dir, f'{video_file}')
    save_video(video_output, output_save_path, fps=10, quality=9, ffmpeg_params=['-pix_fmt', 'yuv420p'])
    print(f"Saved result to {output_save_path}")

    if args.export_target_frames:
        chunk = next(c for c in prepared_manifest['chunks'] if c['video'] == video_file)
        context_count = 2 if previous_frames is not None else 0
        prefix = insert_index + context_count
        expected = len(chunk['reference_ids']) + context_count + len(chunk['target_ids'])
        if insert_index != len(chunk['reference_ids']) or len(video_output) != expected:
            raise ValueError(f'Generated frame count/prefix does not match chunk manifest: {video_file}')
        for local_index, target_id in enumerate(chunk['target_ids']):
            name = f'target_frames/target_{target_id:06d}.png'
            video_output[prefix + local_index].save(Path(args.output_dir) / name)
            target_mapping.append(dict(target_id=target_id, image=name, chunk_id=chunk['chunk_id'],
                                       chunk_frame_index=prefix + local_index,
                                       camera=prepared_manifest['target_cameras'][target_id]))
        (Path(args.output_dir) / 'target_mapping.json').write_text(json.dumps(target_mapping, indent=2) + '\n')

    if not args.independent_chunks and len(video_output) >= 2:
        previous_frames = video_output[-2:]
