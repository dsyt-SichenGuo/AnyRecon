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

浏览器打开 `http://127.0.0.1:8766`。保存第一段前将尺寸设为 896×512，并应用按比例缩放的 K。网页设置前进帧数 n 和环绕帧数 m（默认 5 和 25；n≥1、m≥3、n+m≤1000）。移动到要采集全景的位置，选择起始朝向，**按一次 R（或 T）即可生成并保存整个 Chunk**，无需开始/结束实时录制。

每段先从上一段的最后一个目标相机，直线移动到当前点，按距离均匀生成 n 帧，同时球面插值朝向；n≥2 时包含移动首尾，n=1 时只取到达点。第一段从打开网页时的初始参考相机出发。随后固定相机位置，按 360°/m 步长水平转头，生成 m 帧；第一张环绕图位于第一个角度步长处，最后一张完整转过 360° 回到起始朝向，因此前进终点与第一张环绕图不会重复。没有位置移动时仍可生成原地全景，前进帧的位置相同。全过程不记录时间。

水平面的向上方向默认取初始参考相机的上轴，网页可切换世界 ±X/±Y/±Z；若地面倾斜应先调整。全景起始朝向会被校正为水平并去除滚转；接近竖直的朝向需先手动转向水平方向。保存后视角位于本段闭环终点，可继续移动到下一全景点。按 U 或 Ctrl+Z 撤销整个 Chunk 并返回其前进起点；保存失败时 R 重试同一段，U 放弃。支持继续读取已有轨迹，但新实验建议使用新的 output_dir，避免混入之前的采集结果。完成后 Ctrl+C 停止查看器。相机仍使用 Pi3 世界坐标系和 OpenCV 轴约定。

继续采集已有轨迹有两种方式：直接将 `--output_dir` 指向原来的采集目录，脚本会自动恢复；或用 `--load_targets` 将原轨迹加载到新的输出目录，保留源文件。例如：

```bash
python scripts/select_target_cameras.py \
  --scene outputs/robo_lounge_808_default/reconstruction/scene.npz \
  --load_targets outputs/robo_lounge_808_run02/selected_target_cameras \
  --output_dir outputs/robo_lounge_808_run03/selected_target_cameras \
  --port 8766
```

`--load_targets` 接受目录或 `target_cameras.json` 文件路径。也可以启动空输出目录后，在网页“加载已有目标轨迹”中选择本地 JSON，再点击“加载 JSON 并继续采集”。网页会返回最后一帧，恢复输出尺寸、内参，以及最后一个全景 chunk 的 n/m 和向上方向；之后按 R 在原有目标编号和 chunk 编号之后追加。JSON 包含完整的 chunk 元数据，NPZ 无需手动加载，保存时会自动同步生成。导入会检查场景坐标系和分段边界；输出目录已有另一套轨迹时会拒绝覆盖，请改用空目录。

追加相机后，后续 `prepare_selected_targets.py` 请使用新的准备输出目录，因为原准备结果对应的是旧目标相机文件。

```bash
python scripts/prepare_selected_targets.py \
  --scene "$RUN/reconstruction/scene.npz" \
  --cameras "$RUN/selected_target_cameras/target_cameras.json" \
  --output_dir "$RUN/preprocess" \
  --num_reference 6 --context_chunks

python run_AnyRecon.py \
  --root_dir "$RUN/preprocess" \
  --output_dir "$RUN/generated" \
  --lora_path AnyRecon_full_attention.ckpt \
  --width 896 --height 512 \
  --context_chunks --export_target_frames
```

轨迹文件中的 `chunks` 记录分段边界；打包时自动沿用录制边界，`--chunk_size` 仅用于没有 chunk 信息的旧相机文件。所有 Chunk 使用相同输出尺寸，n+m 越大单组生成显存需求越高。新 Chunk 的 JSON 保存前进/环绕帧数、各自目标 ID、向上轴、环绕角度、前进距离及采样距离；旧距离轨迹仍能读取。NPZ 额外保存 `chunk_offsets/chunk_ids`。撤销会同步更新两份文件。

检查条件 RGB/mask 和参考拼图后再生成。最终图片位于 `generated/target_frames/`，相机对应关系位于 `generated/target_mapping.json`。重新推理使用新的输出目录；准备阶段可按同参数恢复。相邻组需要携带历史输出时，准备与推理两阶段都使用 `--context_chunks`，推理省略 `--independent_chunks`。

## 5. 可配置参考候选池与两种检索方法

`scripts/prepare_selected_targets.py` 一次完成：读取 `scene.npz` 中的完整点云、来源 ID 和参考相机 → 按目标相机渲染 RGB/mask → 导出选中参考 RGB/白色 mask → 按录制的 chunk 边界打包视频、info、manifest 和逐帧相机。无需重跑 Pi3 或加载扩散权重。

| 参数 | 含义 |
| --- | --- |
| `--num_reference 10` | 每段参考图数量；improved 精确选 10 张，original 遵循原代码将 10 作为上限，可能更少 |
| `--reference_method improved` | 改进检索：采样点云粗 z-buffer 近似遮挡、按来源覆盖和新增像素收益选择，首帧为选中参考中贡献最强者 |
| `--reference_method original` | 原项目代码检索：完整点云投影贡献计数，每目标默认取前 10 个来源后汇总排序；保留原代码包含遮挡点的计数方式，参考按原 ID 排序 |
| `--reference_method compare` | 同时生成 original/improved 两套可独立推理的输入；目标 RGB/mask 共用，仅渲染一次 |
| `--reference_ids 0 10 20 30` | 指定候选参考 ID，从 0 开始，对应重建的 `references.json`，不是目标相机 ID |
| `--reference_pool candidates.json` | 从 JSON 数组或 `{"reference_ids": [0, 10, 20]}` 读取 ID；TXT 支持空格/逗号分隔及 `#` 注释 |
| `--reference_pool data/candidates` | 以目录中的 PNG/JPG/JPEG 作为候选；文件名必须与重建参考图唯一对应，不能混入未参与重建的新图片 |
| `--reference_pool_size 200` | 在所选候选池中按参考 ID 均匀减采样为 200 张；省略则不减采样 |
| `--reference_dir data/all_frames` | 重定位原始参考 RGB 目录，按原文件名匹配；仅要求候选池中图片存在。目录型 reference_pool 已提供路径，不同时使用此参数 |

`--reference_ids` 与 `--reference_pool` 二选一；均省略时，候选为重建的全部参考图。改变候选池只改变参考检索，不删减渲染点云；非候选来源的几何仍参与遮挡估计。`scene.npz` 还包含来源 ID、内参及参考路径，不能仅用不含这些信息的裸 PLY 替代。

当前 robo_lounge 示例，在原图路径有效的服务器项目根目录运行；当前相机为 4×30 帧、1280×960：

```bash
BASE=outputs/robo_lounge_808_default
PREP="$BASE/selected120_compare_ref10_pool200"

python -u scripts/prepare_selected_targets.py \
  --scene "$BASE/reconstruction/scene.npz" \
  --cameras "$BASE/selected_target_cameras/target_cameras.json" \
  --output_dir "$PREP" \
  --num_reference 10 \
  --reference_pool_size 200 \
  --reference_method compare
```

要用全部 808 张候选图，去掉 `--reference_pool_size 200`；要指定自己的子集，改用 ID 或候选目录参数。原始图已迁移时增加 `--reference_dir`。本地下载的点云仍引用 `/home/dev/AnyRecon/...` 的原图路径时，必须在服务器运行或提供可用的原图目录；查看器能打开点云不代表参考 RGB 已在本地。

```text
selected120_compare_ref10_pool200/
├── preparation_config.json       # 场景、相机、候选原图和参数，防止错用旧输出
├── reference_pool.json           # 候选 ID、实际路径和来源路径
├── selection_comparison.json     # 相同指标下的平均/最差目标覆盖及逐目标覆盖
├── selection_comparison.html     # 两种方法的参考拼图与覆盖表
├── renders/rgb/target_*.png       # 共用完整点云目标渲染
├── renders/mask/target_*.png
├── original/
│   ├── condition/*.mp4 + *_info.txt
│   ├── mask/*.mp4
│   ├── references/chunk_000/rgb/*.png
│   ├── references/chunk_000/mask/*.png
│   ├── reference_sheets/*.jpg
│   ├── cameras/chunk_000.json     # 输入顺序、RGB/mask 路径、逐帧 c2w/w2c/K
│   ├── manifest.json + target_cameras.json + metadata.csv
│   ├── status/chunk_000.json      # 完成标记及真实渲染 mask 覆盖比例
│   └── generation_command.txt    # 本次输出尺寸和分组模式对应的生成命令
└── improved/                     # 同样结构，可直接用于 run_AnyRecon.py
```

仅运行 original 或 improved 时，上述方法子目录的文件直接位于 output_dir。打包的视频内先放参考 RGB（mask 全白），后放本段目标 RGB/mask；保留录制的目标顺序与 chunk 边界，逐帧相机 JSON 的参考 K 对应缩放后的输出尺寸。

先检查 `selection_comparison.html`、参考图和目标 RGB/mask。覆盖指标采用相同采样与候选池，只是几何代理，不能证明最终生成质量；对原图清晰度、风格或生成细节仍需人工检查。original 如果选中数量更少，比较时也应考虑输入帧数差异。选择一套后可以运行其 `generation_command.txt`；对比最终生成时，两套使用相同 seed、推理步数和分辨率。

添加 `--plan_only` 只检索与写报告，去掉后继续渲染打包；original 检索本身依然遍历全部点，不是采样近似。添加 `--chunk_ids 0` 可先完成第一段，之后去掉继续其余段。同参数重跑可恢复，缺失参考 PNG 或无效视频会重新打包；改变场景、相机、候选池、原图或参数时使用新输出目录。完整点云渲染仍可能占用数十 GB CPU 内存，本步骤不执行 GPU 扩散。

### GPU 参考帧检索

`--render_device cuda` 只控制目标渲染。参考检索需要另外添加 `--retrieval_device cuda`（默认 cpu）。CUDA 检索会将点云上传一次：original 用 float64 分批投影全部点并累计来源计数；improved 在相同采样点和检索尺寸上计算最近深度、可见来源覆盖及覆盖增益选择。候选池以外的点仍参与遮挡，不自动减少点数、候选图数量或参考图数量。原文 top-K 排序和小型统计报告保留在 CPU，improved 的大型覆盖矩阵及增益计算留在 GPU。

例如继续处理新的 25 段轨迹：

```bash
BASE=outputs/robo_lounge_808_run02
python -u scripts/prepare_selected_targets.py \
  --scene outputs/robo_lounge_808_default/reconstruction/scene.npz \
  --cameras "$BASE/selected_target_cameras/target_cameras.json" \
  --output_dir "$BASE/prepared_compare_ref15_pool808" \
  --reference_dir videos/robo_lounge/seq_aria_robo_lounge_6_20261022_175021/all_frames \
  --num_reference 15 --reference_method compare --context_chunks \
  --retrieval_device cuda --retrieval_batch_size 1000000 \
  --render_device cuda --render_batch_size 1000000
```

新版本每完成一个 chunk 就原子保存检索 manifest。中断后保持原参数与输出目录重跑，会复用已完成的参考选择，只补缺失的 chunk；已有完整 CPU 检索结果也会复用，不因切换 CUDA 而强制重算。仍在运行的旧进程需先用 Ctrl+C 停止，再执行新命令。不要同时写入同一个输出目录。改变候选池、采样点数、检索宽度、参考数量等选择参数仍需新目录。

`--retrieval_batch_size` 控制 original 投影临时张量大小，不限制常驻点云和覆盖矩阵显存；30 个目标、808 张候选图、320×240 检索尺寸的布尔覆盖矩阵约 1.73 GiB，另需点云和计算临时空间。GPU 检索缓存在目标渲染之前释放。CUDA 不可用时会报错，不静默切换到 CPU。浮点运算顺序可能让恰好落在投影/深度阈值上的点产生差异，不保证所有场景逐位一致。

RTX 4090 的合成基准（100 万点、6 个目标、808 候选、检索宽 320，预热后且不计首次上传）：improved 投影及选择 CPU 1.621 秒、CUDA 0.085 秒，约 19 倍；original 全点贡献计数 CPU 0.198 秒、CUDA 0.015 秒，约 13 倍。该数据上覆盖掩码、参考选择和原文计数一致。这不是 1.56 亿点真实场景的完整运行耗时。

### GPU 目标点云渲染

条件准备和单视角检查均支持 `--render_device cuda`，默认仍为 `cpu`。CUDA 后端使用已有 PyTorch，不要求编译新扩展；完整点云只上传一次，复用所有目标视角，执行四邻域 bilinear splat、逐像素深度排序和 alpha-over 混合。不会改成最近点渲染或自动减少点数。浮点归约和等深点排序可能造成像素差异，不保证与 CPU 逐位相同。

在能正常执行 `nvidia-smi`、且当前环境 `torch.cuda.is_available()` 为 True 的终端运行：

```bash
BASE=outputs/robo_lounge_808_default
PREP="$BASE/prepared_compare_ref15_pool808"

python -u scripts/prepare_selected_targets.py \
  --scene "$BASE/reconstruction/scene.npz" \
  --cameras "$BASE/selected_target_cameras/target_cameras.json" \
  --reference_dir videos/robo_lounge/seq_aria_robo_lounge_6_20261022_175021/all_frames \
  --output_dir "$PREP" \
  --num_reference 15 \
  --reference_method compare \
  --render_device cuda \
  --render_batch_size 1000000
```

停止旧的 CPU 准备进程后，可以对同一个 PREP 执行以上命令。只要其他参数不变，既有参考检索计划会复用；所请求分段中的 CPU 目标图会用 CUDA 重新渲染，条件视频也会重新打包，避免静默混用。每个目标的 `renders/metadata/target_*.json` 记录后端和耗时；同一后端下已完成的目标可继续复用。

实测 NVIDIA GeForce RTX 4090：100 万点、896×512，在预热后 CPU 约 0.128 秒/帧、CUDA 约 0.011 秒/帧（约 12 倍）；该测试 mask 完全一致，RGB 最大误差为 1 个灰度级。此为小场景测试，不代表 1.56 亿点场景的耗时或显存上限。

CUDA 渲染现在先保留紧凑的可见点投影，再按图像区域自适应分块展开四邻域 splat、排序和混合，避免整帧一次性存储和排序全部 splat。跨图块的 bilinear 贡献分别归入对应像素，不减少点数。`--render_batch_size` 控制投影临时数组；新增 `--render_max_splats` 控制每块排序的 splat 数量预算，默认 32000000。排序 OOM 时会自动继续细分图块。常驻点云、紧凑投影和排序工作空间仍需显存，该预算不是整个进程的显存上限；极端单像素密集场景无法继续细分时会明确报错。

例如大点云遇到排序 OOM 后，停止旧进程，保持原输入及输出目录和其他参数不变，增加 `--render_batch_size 500000 --render_max_splats 16000000` 后重跑。已有参考检索计划和已完成的 CUDA 目标图会复用，从缺失目标继续。需要更低排序显存时可进一步减小 render_max_splats，代价是更多图块和更长耗时。若点云及紧凑投影本身也无法放入 GPU，则使用更大显存或 `--render_device cpu`。

目标渲染和参考检索分别由 render_device / retrieval_device 控制；PNG/视频编码及磁盘读取仍在 CPU。`--plan_only` 不渲染目标，但选择 `--retrieval_device cuda` 时仍需 GPU。

也可先独立检查一个目标，输出目录须为空或不存在：

```bash
python scripts/render_selected_targets.py \
  --scene "$BASE/reconstruction/scene.npz" \
  --cameras "$BASE/selected_target_cameras/target_cameras.json" \
  --frame_ids 0 \
  --output_dir "$BASE/check_cuda_target0" \
  --render_device cuda
```

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
