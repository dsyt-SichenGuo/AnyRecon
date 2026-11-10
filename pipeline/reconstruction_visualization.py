"""Offline, interactive visualization in the cached Pi3 world coordinate frame."""
import base64
from html import escape
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image


def camera_frustum(c2w, intrinsics, image_size, depth):
    """Five world-space vertices: camera center followed by four image corners."""
    width, height = image_size
    pixels = np.array([[0, 0, 1], [width - 1, 0, 1],
                       [width - 1, height - 1, 1], [0, height - 1, 1]], dtype=np.float64)
    rays = (np.linalg.inv(intrinsics) @ pixels.T).T * depth
    corners = (c2w[:3, :3] @ rays.T).T + c2w[:3, 3]
    return np.vstack([c2w[:3, 3], corners])


def write_scene_viewer(scene_path, output_path, max_points=200000):
    import plotly.graph_objects as go

    if max_points <= 0:
        raise ValueError('max_points must be positive')
    with np.load(scene_path, allow_pickle=False) as scene:
        total = len(scene['points'])
        indices = np.linspace(0, total - 1, min(total, max_points), dtype=np.int64)
        points, colors = scene['points'][indices], scene['colors'][indices]
        sources = scene['source_indices'][indices]
        count = int(scene['num_reference_frames'])
        poses, intrinsics = scene['input_c2w'][:count], scene['intrinsics_reconstruction'][:count]
        image_size = scene['image_size_reconstruction'].tolist()
        frame_sources = scene['frame_sources'][:count].tolist()
        reference_rgb = scene['reference_rgb'][:count]
    rgb = [f'rgb({r},{g},{b})' for r, g, b in colors]
    cloud = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], mode='markers', name='彩色点云',
        marker=dict(size=1.5, color=rgb, opacity=1), customdata=sources,
        hovertemplate='X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<br>参考图 ID=%{customdata}<extra></extra>',
    )
    span = np.linalg.norm(np.percentile(points, 98, axis=0) - np.percentile(points, 2, axis=0))
    frustum_depth = max(float(span) * 0.025, 1e-5)
    edges = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
    lines = [[], [], []]
    for pose, K in zip(poses, intrinsics):
        vertices = camera_frustum(pose, K, image_size, frustum_depth)
        for a, b in edges:
            for axis in range(3):
                lines[axis].extend([vertices[a, axis], vertices[b, axis], None])
    frustums = go.Scatter3d(x=lines[0], y=lines[1], z=lines[2], mode='lines',
                          line=dict(color='#ffba66', width=3), name='参考相机朝向', hoverinfo='skip')
    centers = poses[:, :3, 3]
    cameras = go.Scatter3d(
        x=centers[:, 0], y=centers[:, 1], z=centers[:, 2], mode='markers+text',
        text=[str(i) for i in range(count)], textposition='top center',
        marker=dict(size=5, color='#65d7ff'), name='参考相机中心',
        customdata=[[i, Path(source).name] for i, source in enumerate(frame_sources)],
        hovertemplate='参考图 %{customdata[0]}<br>%{customdata[1]}<br>X=%{x:.4f}<br>Y=%{y:.4f}<br>Z=%{z:.4f}<extra></extra>',
    )
    fig = go.Figure([cloud, frustums, cameras])
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#101820', margin=dict(l=0, r=0, b=0, t=45),
        scene=dict(aspectmode='data', dragmode='orbit',
                   xaxis_title='World X', yaxis_title='World Y', zaxis_title='World Z',
                   camera=dict(eye=dict(x=1.3, y=1.3, z=0.9))),
        legend=dict(x=0.01, y=0.98), uirevision='pi3-reference-scene',
        updatemenus=[dict(type='buttons', direction='right', x=0, y=1.06,
                         buttons=[dict(label='RGB 颜色', method='restyle', args=[{'marker.color': [rgb], 'marker.showscale': False}, [0]]),
                                  dict(label='按参考帧着色', method='restyle',
                                       args=[{'marker.color': [sources.tolist()], 'marker.colorscale': 'Turbo',
                                              'marker.cmin': 0, 'marker.cmax': max(1, count - 1),
                                              'marker.showscale': True}, [0]])])],
    )
    cards = []
    for i, (image, source) in enumerate(zip(reference_rgb[:100], frame_sources[:100])):
        thumbnail = Image.fromarray(image)
        thumbnail.thumbnail((240, 150))
        buffer = BytesIO()
        thumbnail.save(buffer, format='JPEG', quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        cards.append(f'<article><strong>参考图 {i}</strong><span>{escape(Path(source).name)}</span>'
                     f'<img alt="参考图 {i}" src="data:image/jpeg;base64,{encoded}"></article>')
    plot = fig.to_html(full_html=False, include_plotlyjs=True,
                       config=dict(displaylogo=False, scrollZoom=True, responsive=True))
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pi3 参考图点云</title>
<style>html,body{{margin:0;height:100%;font:14px system-ui,sans-serif;background:#101820;color:#e5edf3}}
main{{display:grid;grid-template-columns:minmax(0,1fr) 300px;height:100vh}}#scene{{height:100vh;min-width:0}}#scene>div{{height:100%}}
aside{{overflow:auto;padding:20px;background:#192630;border-left:1px solid #334550}}h1{{font-size:21px;margin-top:0}}
p{{line-height:1.7;color:#b8cbd6}}.stats{{color:#65d7ff;font-weight:600}}article{{padding:12px 0;border-top:1px solid #334550}}
article span{{display:block;color:#b8cbd6;margin:5px 0;overflow-wrap:anywhere}}img{{max-width:100%;border-radius:6px}}
@media(max-width:850px){{main{{grid-template-columns:1fr}}#scene{{height:70vh}}aside{{overflow:visible}}}}
</style></head><body><main><div id="scene">{plot}</div><aside>
<h1>Pi3 参考图重建</h1><p class="stats">{count} 张参考图 · 完整点云 {total:,} 点<br>当前显示 {len(indices):,} 点</p>
<p>拖动旋转，滚轮缩放；工具栏可切换平移。点击图例可隐藏点云或相机。悬停可查看坐标和参考图 ID。</p>
<p>蓝色标记为相机中心，橙色线框表示朝向。世界坐标未经移动或缩放；单位为相对重建尺度，不是米。相机采用 OpenCV：X 右、Y 下、Z 前。</p>
<p>视锥大小仅用于显示。原始位姿、内参和完整点云保存在 scene.npz / cameras.json。本页用于检查重建，尚不导出新相机轨迹。</p>
<h2>参考图</h2>{''.join(cards)}{'<p>仅显示前100张参考图缩略图。</p>' if count > 100 else ''}
</aside></main></body></html>'''
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    print(f'Saved offline point-cloud viewer: {output}')
    return output
