# AnyRecon：原项目运行说明

AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model。
[原项目](https://github.com/OpenImagingLab/AnyRecon) · [项目主页](https://yutian10.github.io/AnyRecon/) · [模型](https://huggingface.co/Yutian10/AnyRecon)

本地工作目录保留三份使用文档；Git 仓库仅跟踪本 README，其余两份为本地说明：

| 文档 | 内容 |
| --- | --- |
| [README.md](README.md) | 原项目的输入组织、Pi3 条件渲染、AnyRecon 推理 |
| `TARGET_CAMERA_WORKFLOW.md`（本地） | 参考图重建 → 手选相机轨迹 → 检索与渲染 → 扩散生成 → 目标图片验证 |
| `ENVIRONMENT_SETUP.md`（本地） | egohp 服务器接入、源码与数据迁移、环境和权重配置、停机 |

本目录是源码副本，已包含 `Pi3/` 和 `diffsynth/`，但未携带权重、输入数据、实验输出或原仓库示例。旧文档里提到的 Aria 数据、400 帧点云、24/60 个已选相机及历史生成结果，不表示当前目录已经存在这些文件。先按环境文档准备服务器与模型。以下命令均在项目根目录执行。

## 1. 原项目的输入与行为

原版入口 `run_pi3.py` 接收视频，或图片目录；图片可直接放在目录下，也可放在其 `images_undistort/` 子目录中。图片按文件名排序，建议使用补零编号。

- 前 `num_cond_frames` 帧是参考图，提供点云。
- 后续帧是目标图，只用于联合估计目标位姿和内参，不提供点云。
- Pi3 会联合处理全部输入；这些目标 RGB 参与几何估计，不能称为“仅参考图重建”。
- 条件视频由前置参考 RGB 和目标位姿下的点云渲染组成；mask 中白色代表有效条件。

例如使用 6 张参考图和 35 张目标图：

```text
data/original_scene/images_undistort/
├── 000000.png ... 000005.png   # 6 张参考
└── 000006.png ... 000040.png   # 35 张目标
```

当前 `run_pi3.py` 保留原版行为：输出 896×512，每段固定 35 张目标，检索最多 10 张参考，Pi3 默认像素预算 255000。**不足 35 张的尾段会被跳过；目标总数应为 35 的整数倍。**该入口没有 `--chunk_size`、`--pixel_limit`、`--width/--height`、`--reconstruction_only` 等扩展命令行参数，也不会自动导出自选相机流程所需的 `scene.npz`。

## 2. Pi3 几何估计与条件渲染

激活环境后，在项目根目录设置路径。云端环境的激活方式见环境文档。

```bash
export PYTHONDONTWRITEBYTECODE=1
RUN=outputs/original_scene_run01
mkdir -p "$RUN"

python -u run_pi3.py \
  --base_scene_dir data/original_scene \
  --ckpt Pi3/model.safetensors \
  --device cuda \
  --num_cond_frames 6 \
  --use_retrieval \
  --seed 42 \
  --output_dir "$RUN/preprocess" \
  > "$RUN/pi3.log" 2>&1
```

也可将 `--base_scene_dir` 指向实际视频文件。默认不进行检索，添加 `--use_retrieval` 后按几何贡献为各段选择参考图。

输出包括 `preprocess/condition/*.mp4`、同名 `_info.txt`、`preprocess/mask/*.mp4`，以及 `train/`、`cameras/`、`metadata.csv`。继续前检查日志、RGB/mask 和 info：

```bash
cat "$RUN"/preprocess/condition/*_info.txt
```

上述示例应有一段，`Render Frame Count: 35`。`cameras/` 中的原版来源记录不能替代完整重建缓存。

## 3. AnyRecon 扩散推理

```bash
python -u run_AnyRecon.py \
  --root_dir "$RUN/preprocess" \
  --output_dir "$RUN/generated" \
  --lora_path AnyRecon_full_attention.ckpt \
  --width 896 --height 512 \
  --num_inference_steps 4 \
  --seed 1 \
  > "$RUN/anyrecon.log" 2>&1
```

使用 full-attention 权重时省略 `--is_block`。block 模式会向下截取到 8 的倍数，不适用于这里的默认示例。

输出是各段同名 MP4，包含前置参考帧；第二段起还会在参考之后插入前一段最后两张生成帧及白色 mask。因此不能直接拼接各段 MP4 作为纯目标视频。只有一段时，去掉 info 指定数量的前置参考帧后才是目标结果。

`--export_target_frames` 需要自选相机打包脚本产生的专用 manifest，不能直接用于本节原版预处理结果。需要纯目标 PNG 和逐帧相机映射时，请使用 下节的自选相机 pipeline；原版入口仅输出包含辅助帧的分段 MP4。

`test.sh` / `run_pi3.sh` 仍指向上游示例路径；当前源码副本未包含这些示例，应使用上面的实际输入命令。原项目报告约 40 帧、896×512 推理需要约 45GB 显存，实际占用以所用 GPU 和输入为准。代码未实现多 GPU 分布式推理。

## 4. 自定义轨迹入口

要由自己选择相机，无需制造目标占位图或修改原版 `process_scene`。从项目根目录依次执行以下步骤。先用 Python 3.10 创建环境，安装 Torch 2.4.1/cu118 和 `requirements-runtime.txt`，确保 FFmpeg 支持 libx264；使用内置 diffsynth，不额外安装 PyPI 版本。

```bash
python scripts/download_weights.py
python scripts/check_environment.py

RUN=outputs/custom_camera_run01
mkdir -p "$RUN"

# data/reference_rgb/ 中只放要用于重建的参考图片，至少 6 张。
python run_pi3_reconstruct.py \
  --reference_dir data/reference_rgb \
  --ckpt Pi3/model.safetensors \
  --output_dir "$RUN/reconstruction"

python scripts/select_target_cameras.py \
  --scene "$RUN/reconstruction/scene.npz" \
  --output_dir "$RUN/selected_target_cameras" \
  --port 8766
```

浏览器打开 `http://127.0.0.1:8766`。保存第一帧前将尺寸设为 896×512，并应用按比例缩放的 K；按最终顺序保存目标相机，完成后 Ctrl+C 停止查看器。相机使用 Pi3 世界坐标系和 OpenCV 轴约定，保存的视角不会自动插值。远程节点没有浏览器访问方式时，可下载同一份 scene 缓存到本地选相机，再上传目标相机 JSON/NPZ；节点上的原始参考图路径需保持有效。

```bash
python scripts/prepare_selected_targets.py \
  --scene "$RUN/reconstruction/scene.npz" \
  --cameras "$RUN/selected_target_cameras/target_cameras.json" \
  --output_dir "$RUN/preprocess" \
  --chunk_size 4 --num_reference 6

python run_AnyRecon.py \
  --root_dir "$RUN/preprocess" \
  --output_dir "$RUN/generated" \
  --lora_path AnyRecon_full_attention.ckpt \
  --width 896 --height 512 \
  --independent_chunks --export_target_frames
```

检查条件 RGB/mask 和参考拼图后再生成。最终图片位于 `generated/target_frames/`，相机对应关系位于 `generated/target_mapping.json`。重新推理使用新的输出目录；准备阶段可按同参数恢复。相邻组需要携带历史输出时，准备与推理两阶段都使用 `--context_chunks`，推理省略 `--independent_chunks`。

## 来源、致谢与引用

主项目基于 [Wan2.1](https://github.com/Wan-Video/Wan2.1)、[DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) 和 [π³](https://github.com/yyfz/Pi3/)。Pi3 的上游说明及 Pi3X 用法见其官方仓库；本 pipeline 使用 Pi3，权重为 `Pi3/model.safetensors`。Pi3 代码许可证保留于 [Pi3/LICENSE](Pi3/LICENSE)；上游模型权重声明为 CC BY-NC 4.0，使用范围以各模型原始许可证为准。

```bibtex
@article{chen2026anyrecon,
  title={AnyRecon: Arbitrary-View 3D Reconstruction with Video Diffusion Model},
  author={Chen, Yutian and Guo, Shi and Jin, Renbiao and Yang, Tianshuo and Cai, Xin and Luo, Yawen and Yang, Mingxin and Yu, Mulin and Xu, Linning and Xue, Tianfan},
  journal={arXiv preprint arXiv:2604.19747},
  year={2026}
}

@article{wang2025pi,
  title={$$\backslash$pi\^{} 3$: Permutation-Equivariant Visual Geometry Learning},
  author={Wang, Yifan and Zhou, Jianjun and Zhu, Haoyi and Chang, Wenzheng and Zhou, Yang and Li, Zizun and Chen, Junyi and Pang, Jiangmiao and Shen, Chunhua and He, Tong},
  journal={arXiv preprint arXiv:2507.13347},
  year={2025}
}
```
