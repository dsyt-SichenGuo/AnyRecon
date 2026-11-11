"""Optional CUDA bilinear splats with depth-ordered alpha-over compositing.

The CPU backend remains run_pi3's original renderer. The tensor implementation
runs on CPU in regression tests as well as CUDA in production; no extension build
or point-cloud subsampling is required. Equal-depth ties and floating-point
reduction order can cause small differences from the original NumPy/Numba path.
"""
import numpy as np
import torch


class PointCloudRenderer:
    def __init__(self, points, colors, device='cpu', batch_size=1000000):
        if batch_size <= 0:
            raise ValueError('render_batch_size must be positive')
        if device not in ('cpu', 'cuda'):
            raise ValueError('render_device must be cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this process; check nvidia-smi and the PyTorch CUDA installation')
        self.device = device
        self.batch_size = batch_size
        self.backend = 'cpu_original_v1' if device == 'cpu' else 'cuda_alpha_over_v1'
        self.impl = (TensorSplatRenderer(points, colors, 'cuda', batch_size) if device == 'cuda'
                     else (points, colors))

    def render(self, camera):
        if self.device == 'cpu':
            from run_pi3 import render_point_cloud_to_memory
            points, colors = self.impl
            frames, masks, _ = render_point_cloud_to_memory(points, colors,
                np.asarray([camera['w2c']]), np.asarray([camera['K']]), tuple(camera['image_size']))
            return frames[0], masks[0]
        try:
            return self.impl.render(camera)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError('CUDA splat rendering ran out of memory. Projection batch size controls only '
                               'projection temporaries; depth sorting still retains every visible splat. '
                               'Use a GPU with more free VRAM or --render_device cpu; no points were dropped.') from error

    def close(self):
        self.impl = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()


class TensorSplatRenderer:
    """Project in batches, sort splats per pixel, then composite all depths in parallel."""
    def __init__(self, points, colors, device='cuda', batch_size=1000000):
        if batch_size <= 0:
            raise ValueError('render_batch_size must be positive')
        if len(points.shape) != 2 or points.shape[1] != 3 or colors.shape != points.shape or len(points) == 0:
            raise ValueError('Expected nonempty points[P,3] and matching colors[P,3]')
        colors = np.asarray(colors)
        if colors.dtype != np.uint8:
            colors = colors.astype(np.float32)
            if colors.max() <= 1:
                colors = colors * 255
            colors = np.clip(colors, 0, 255).astype(np.uint8)
        self.device = torch.device(device)
        self.batch_size = batch_size
        # Upload the full cloud once and reuse it for every target camera.
        self.points = torch.as_tensor(np.asarray(points), device=self.device)
        self.colors = torch.as_tensor(colors, device=self.device)
        self.index_dtype = torch.int32 if len(points) < 2**31 else torch.int64

    @torch.no_grad()
    def _project(self, camera):
        width, height = camera['image_size']
        E = torch.as_tensor(camera['w2c'], dtype=torch.float64, device=self.device)
        K = torch.as_tensor(camera['K'], dtype=torch.float64, device=self.device)
        for start in range(0, len(self.points), self.batch_size):
            end = min(len(self.points), start + self.batch_size)
            xyz = self.points[start:end].to(torch.float64) @ E[:3, :3].T + E[:3, 3]
            keep = (xyz[:, 2] > 0.1) & (self.colors[start:end].sum(dim=1) > 0)
            ids = torch.arange(start, end, dtype=self.index_dtype, device=self.device)[keep]
            projected = xyz[keep] @ K.T
            depth = projected[:, 2]
            uv = projected[:, :2] / (depth[:, None] + 1e-8)
            keep = ((uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5)
                    & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5))
            # Match the original renderer's float32 coordinates at the splat stage.
            uv, depth, ids = uv[keep].to(torch.float32), depth[keep], ids[keep]
            floors = torch.floor(uv).to(torch.int64)
            keep = ((floors[:, 0] >= 0) & (floors[:, 0] < width - 1)
                    & (floors[:, 1] >= 0) & (floors[:, 1] < height - 1))
            uv, floors, depth, ids = uv[keep], floors[keep], depth[keep], ids[keep]
            fractions = uv.to(torch.float64) - floors
            u, v = fractions[:, 0], fractions[:, 1]
            for dx, dy, weights in [(0, 0, (1-u)*(1-v)), (1, 0, u*(1-v)),
                                    (0, 1, (1-u)*v), (1, 1, u*v)]:
                keep = weights > 1e-8
                pixels = ((floors[:, 1] + dy) * width + floors[:, 0] + dx).to(torch.int32)
                yield pixels[keep], depth[keep], weights[keep], ids[keep]

    @torch.no_grad()
    def render(self, camera):
        width, height = camera['image_size']
        if width <= 0 or height <= 0 or width*height >= 2**31:
            raise ValueError('Invalid render dimensions')
        pixel_parts, depth_parts, weight_parts, point_parts = [], [], [], []
        for pixels, depth, weights, ids in self._project(camera):
            if pixels.numel():
                pixel_parts.append(pixels)
                depth_parts.append(depth)
                weight_parts.append(weights)
                point_parts.append(ids)
        if not pixel_parts:
            blank = np.zeros((height, width, 3), np.uint8)
            return blank, blank.copy()
        pixels = torch.cat(pixel_parts)
        depth = torch.cat(depth_parts)
        weights = torch.cat(weight_parts)
        ids = torch.cat(point_parts)
        del pixel_parts, depth_parts, weight_parts, point_parts
        # Near-to-far within each pixel. Stable pixel sorting preserves depth order.
        order = torch.argsort(depth, stable=True)
        del depth
        pixel_order = torch.argsort(pixels[order], stable=True)
        order = order[pixel_order]
        del pixel_order
        pixels, weights, ids = pixels[order], weights[order], ids[order]
        del order
        # Alpha-over from far to near is equivalent to front-to-back weighted sums:
        # weight_i = alpha_i * product(1-alpha_j) for all nearer j.
        # Log products avoid global underflow across long pixel groups. Epsilon only
        # avoids log(0) at fully opaque samples; its residual is below RGB precision.
        logs = torch.log1p(-weights.clamp(max=1-1e-12))
        prefix = torch.cumsum(logs, dim=0) - logs
        del logs
        starts = torch.cat((torch.zeros(1, dtype=torch.int64, device=self.device),
                            torch.nonzero(pixels[1:] != pixels[:-1]).flatten() + 1))
        base = torch.zeros(width*height, dtype=torch.float64, device=self.device)
        base[pixels[starts]] = prefix[starts]
        prefix -= base[pixels]
        contribution = prefix.exp_().mul_(weights)
        del weights, base, starts, prefix
        alpha = torch.zeros(width*height, dtype=torch.float64, device=self.device)
        alpha.index_add_(0, pixels, contribution)
        rgb = torch.empty((width*height, 3), dtype=torch.uint8, device=self.device)
        for channel in range(3):
            values = self.colors[ids, channel].to(torch.float64).mul_(contribution)
            output = torch.zeros_like(alpha)
            output.index_add_(0, pixels, values)
            output.div_(alpha.clamp_min(1e-6)).clamp_(0, 255)
            rgb[:, channel] = output.to(torch.uint8)
            del values, output
        mask = ((alpha > 0.01).to(torch.uint8) * 255).reshape(height, width).cpu().numpy()
        # All pipeline image/video I/O uses BGR, matching run_pi3.
        bgr = rgb.reshape(height, width, 3).flip(-1).cpu().numpy()
        return bgr, np.repeat(mask[:, :, None], 3, axis=2)
