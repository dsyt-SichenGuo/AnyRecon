"""Optional CUDA bilinear splats with depth-ordered alpha-over compositing.

The CPU backend remains run_pi3's original renderer. The tensor implementation
runs on CPU in regression tests as well as CUDA in production; no extension build
or point-cloud subsampling is required. Equal-depth ties and floating-point
reduction order can cause small differences from the original NumPy/Numba path.
"""
import numpy as np
import torch


class PointCloudRenderer:
    def __init__(self, points, colors, device='cpu', batch_size=1000000, max_splats=32000000):
        if batch_size <= 0 or max_splats <= 0:
            raise ValueError('render_batch_size and render_max_splats must be positive')
        if device not in ('cpu', 'cuda'):
            raise ValueError('render_device must be cpu or cuda')
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this process; check nvidia-smi and the PyTorch CUDA installation')
        self.device = device
        self.batch_size = batch_size
        self.backend = 'cpu_original_v1' if device == 'cpu' else 'cuda_alpha_over_v1'
        self.impl = (TensorSplatRenderer(points, colors, 'cuda', batch_size, max_splats) if device == 'cuda'
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
            raise RuntimeError('CUDA tiled rendering ran out of memory. Free GPU memory, lower '
                               '--render_batch_size / --render_max_splats, or use --render_device cpu. '
                               'The full cloud and compact visible projections still need GPU memory; '
                               'no points were dropped.') from error

    def close(self):
        self.impl = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()


class TensorSplatRenderer:
    """Project once into compact points, then sort/composite bounded image tiles."""
    def __init__(self, points, colors, device='cuda', batch_size=1000000, max_splats=32000000):
        if batch_size <= 0 or max_splats <= 0:
            raise ValueError('render_batch_size and render_max_splats must be positive')
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
        self.max_splats = max_splats
        self.last_tile_count = 0
        self.last_max_tile_splats = 0
        # Upload the full cloud once and reuse it for every target camera.
        self.points = torch.as_tensor(np.asarray(points), device=self.device)
        self.colors = torch.as_tensor(colors, device=self.device)
        self.index_dtype = torch.int32 if len(points) < 2**31 else torch.int64

    @torch.no_grad()
    def _project_points(self, camera):
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
            yield uv, depth, ids

    @torch.no_grad()
    def render(self, camera):
        width, height = camera['image_size']
        if width <= 0 or height <= 0 or width*height >= 2**31:
            raise ValueError('Invalid render dimensions')
        uv_parts, depth_parts, point_parts = [], [], []
        for uv, depth, ids in self._project_points(camera):
            if ids.numel():
                uv_parts.append(uv)
                depth_parts.append(depth)
                point_parts.append(ids)
        rgb = np.zeros((height, width, 3), np.uint8)
        mask = np.zeros_like(rgb)
        self.last_tile_count = self.last_max_tile_splats = 0
        if not uv_parts:
            return rgb, mask
        # Three compact arrays, not four fully expanded splat arrays for the view.
        uv = torch.cat(uv_parts)
        depth = torch.cat(depth_parts)
        ids = torch.cat(point_parts)
        del uv_parts, depth_parts, point_parts
        self._render_region(uv, depth, ids, (0, 0, width, height), rgb, mask)
        return rgb, mask

    def _split_region(self, uv, depth, ids, bounds, rgb, mask):
        x0, y0, x1, y1 = bounds
        if x1-x0 >= y1-y0 and x1-x0 > 1:
            middle = (x0+x1)//2
            regions = [(x0, y0, middle, y1), (middle, y0, x1, y1)]
        elif y1-y0 > 1:
            middle = (y0+y1)//2
            regions = [(x0, y0, x1, middle), (x0, middle, x1, y1)]
        else:
            raise RuntimeError('Too many splats in a single pixel for the render_max_splats budget; '
                               'increase that budget if VRAM permits, or use --render_device cpu. No points were dropped.')
        for region in regions:
            self._render_region(uv, depth, ids, region, rgb, mask)

    def _render_region(self, uv, depth, ids, bounds, rgb, mask):
        x0, y0, x1, y1 = bounds
        # Include points whose bilinear neighbor crosses into this tile.
        inside = ((uv[:, 0] >= x0-1) & (uv[:, 0] < x1)
                  & (uv[:, 1] >= y0-1) & (uv[:, 1] < y1))
        count = int(inside.sum().item())
        if not count:
            return
        estimate = count * min(2, x1-x0) * min(2, y1-y0)
        if estimate > self.max_splats:
            del inside
            self._split_region(uv, depth, ids, bounds, rgb, mask)
            return
        try:
            tile_rgb, tile_mask = self._render_tile(uv, depth, ids, inside, bounds)
        except torch.cuda.OutOfMemoryError:
            # Exit the exception scope before retrying so tensor tracebacks release
            # failed sort buffers. Finer tiles retain all contributions.
            pass
        else:
            rgb[y0:y1, x0:x1] = tile_rgb
            mask[y0:y1, x0:x1] = tile_mask
            self.last_tile_count += 1
            return
        del inside
        torch.cuda.empty_cache()
        self._split_region(uv, depth, ids, bounds, rgb, mask)

    def _render_tile(self, all_uv, all_depth, all_ids, inside, bounds):
        x0, y0, x1, y1 = bounds
        width, height = x1-x0, y1-y0
        uv, point_depth, point_ids = all_uv[inside], all_depth[inside], all_ids[inside]
        floors = torch.floor(uv).to(torch.int64)
        fractions = uv.to(torch.float64) - floors
        u, v = fractions[:, 0], fractions[:, 1]
        pixel_parts, depth_parts, weight_parts, point_parts = [], [], [], []
        for dx, dy, weights in [(0, 0, (1-u)*(1-v)), (1, 0, u*(1-v)),
                                (0, 1, (1-u)*v), (1, 1, u*v)]:
            px, py = floors[:, 0]+dx, floors[:, 1]+dy
            keep = ((weights > 1e-8) & (px >= x0) & (px < x1) & (py >= y0) & (py < y1))
            pixel_parts.append(((py[keep]-y0)*width+px[keep]-x0).to(torch.int32))
            depth_parts.append(point_depth[keep])
            weight_parts.append(weights[keep])
            point_parts.append(point_ids[keep])
        pixels, depth = torch.cat(pixel_parts), torch.cat(depth_parts)
        weights, ids = torch.cat(weight_parts), torch.cat(point_parts)
        del pixel_parts, depth_parts, weight_parts, point_parts
        del uv, point_depth, point_ids, floors, fractions, u, v, px, py, keep
        self.last_max_tile_splats = max(self.last_max_tile_splats, len(pixels))
        if not pixels.numel():
            blank = np.zeros((height, width, 3), np.uint8)
            return blank, blank.copy()
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
