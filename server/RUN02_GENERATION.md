# run02：服务器端 AnyRecon 视频生成

本次输入已准备完成：**improved 参考选择、17 个连续 chunk，每段 15 张参考帧 + 30 张目标帧，共 510 个目标帧，1280×960**。直接生成，不再重建点云、检索参考图或渲染条件图。

## 1. 本地打包并上传

在本地项目根目录执行：

```bash
tar -C outputs/robo_lounge_808_run02/prepared_compare_ref15_pool808 \
  -czf /tmp/robo_lounge_run02_improved.tar.gz improved
```

通过服务器平台上传整个压缩包，以下假设上传至 `/home/dev/robo_lounge_run02_improved.tar.gz`。不用上传点云或原始 808 张图片。

## 2. 服务器生成

假设项目位于 `/home/dev/AnyRecon`，已使用自己 fork 的 `ego_generation` 分支，环境和权重均已配置。路径或环境名不同时自行替换。

```bash
conda activate anyrecon
cd /home/dev/AnyRecon
export PYTHONNOUSERSITE=1

BASE=/home/dev/AnyRecon/outputs/robo_lounge_808_run02
mkdir -p "$BASE/inputs"
tar -xzf /home/dev/robo_lounge_run02_improved.tar.gz -C "$BASE/inputs"

python -u run_AnyRecon.py \
  --root_dir "$BASE/inputs/improved" \
  --output_dir "$BASE/generated_improved_context" \
  --lora_path AnyRecon_full_attention.ckpt \
  --width 1280 --height 960 \
  --num_inference_steps 4 \
  --seed 42 \
  --context_chunks \
  --export_target_frames \
  > "$BASE/anyrecon.log" 2>&1
```

**同一次进程顺序生成全部 17 段**，后续段自动接收上一段最后两张生成图。不要使用 `--independent_chunks`，不要分开启动各个 chunk。长任务使用服务器平台的后台任务功能；重跑请换新的生成输出目录。

## 3. 合成最终视频

生成结束后，在同一终端执行：

```bash
ffmpeg -n -framerate 10 -start_number 0 \
  -i "$BASE/generated_improved_context/target_frames/target_%06d.png" \
  -frames:v 510 -c:v libx264 -crf 18 -pix_fmt yuv420p \
  "$BASE/anyrecon_run02_510f_10fps.mp4"
```

最终视频 **510 帧、10 FPS，约 51 秒**，不含参考帧和历史上下文。轨迹按距离采样，10 FPS 仅决定播放速度。

下载最终 MP4；需要保留逐帧数据时，同时保存 `generated_improved_context/target_frames/`、`target_mapping.json` 和 `target_cameras.json`。日志位于 `$BASE/anyrecon.log`。保存结果后停止本次使用的计算节点。
