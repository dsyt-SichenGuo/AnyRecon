"""CUDA reference retrieval; original full-cloud counts and sampled visible coverage."""
import numpy as np
import torch


class CudaReferenceRetriever:
    def __init__(self, sample, sources, num_references, candidate_ids,
                 points=None, colors=None, full_sources=None, batch_size=1000000):
        if not torch.cuda.is_available():
            raise ValueError('CUDA reference retrieval requires an available CUDA device')
        self.sample = torch.as_tensor(sample, dtype=torch.float32, device='cuda')
        self.sources = torch.as_tensor(sources, dtype=torch.int64, device='cuda')
        self.num_references = num_references
        self.candidates = torch.as_tensor(candidate_ids, dtype=torch.int64, device='cuda')
        self.mapping = torch.full((num_references,), -1, dtype=torch.int64, device='cuda')
        self.mapping[self.candidates] = torch.arange(len(candidate_ids), device='cuda')
        self.batch_size = batch_size
        self.points = self.full_sources = self.nonblack = None
        if points is not None:
            # Keep source precision; project in float64 to match original_source_counts.
            self.points = torch.as_tensor(points, device='cuda')
            self.full_sources = torch.as_tensor(full_sources, dtype=torch.int64, device='cuda')
            self.nonblack = torch.as_tensor(np.any(colors, axis=1), device='cuda')

    @torch.inference_mode()
    def coverage(self, camera, width=320, depth_tolerance=0.01):
        out_width, out_height = camera['image_size']
        height = max(1, round(width * out_height / out_width))
        K = torch.tensor(camera['K'], dtype=torch.float32, device='cuda')
        K[0] *= width / out_width
        K[1] *= height / out_height
        E = torch.tensor(camera['w2c'], dtype=torch.float32, device='cuda')
        xyz = self.sample @ E[:3, :3].T + E[:3, 3]
        front = xyz[:, 2] > 0.1
        xyz, sources = xyz[front], self.sources[front]
        projected = xyz @ K.T
        pixels = torch.floor(projected[:, :2] / projected[:, 2:3] + 0.5).to(torch.int64)
        inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        pixels, sources, depth = pixels[inside], sources[inside], xyz[inside, 2]
        flat = pixels[:, 1] * width + pixels[:, 0]
        nearest = torch.full((width * height,), torch.inf, device='cuda')
        nearest.scatter_reduce_(0, flat, depth, reduce='amin', include_self=True)
        visible = depth <= nearest[flat] * (1 + depth_tolerance) + 1e-6
        selected_sources = self.mapping[sources[visible]]
        eligible = selected_sources >= 0
        masks = torch.zeros((len(self.candidates), width * height), dtype=torch.bool, device='cuda')
        # Duplicate source/pixel pairs always write the same value.
        indices = selected_sources[eligible] * (width * height) + flat[visible][eligible]
        masks.view(-1).index_fill_(0, indices, True)
        return masks

    @torch.inference_mode()
    def chunk_coverage(self, cameras, width=320, depth_tolerance=0.01):
        result = None
        for i, camera in enumerate(cameras):
            row = self.coverage(camera, width, depth_tolerance)
            if result is None:
                result = torch.empty((len(cameras), *row.shape), dtype=torch.bool, device='cuda')
            result[i].copy_(row)
        return result

    @torch.inference_mode()
    def original_counts(self, camera):
        if self.points is None:
            raise ValueError('Original retrieval requires the full point cloud')
        E = torch.tensor(camera['w2c'], dtype=torch.float64, device='cuda')
        K = torch.tensor(camera['K'], dtype=torch.float64, device='cuda')
        width, height = camera['image_size']
        counts = torch.zeros(self.num_references, dtype=torch.int64, device='cuda')
        for start in range(0, len(self.points), self.batch_size):
            end = start + self.batch_size
            xyz = self.points[start:end].to(torch.float64) @ E[:3, :3].T + E[:3, 3]
            keep = (xyz[:, 2] > 0.1) & self.nonblack[start:end]
            projected = xyz[keep] @ K.T
            uv = projected[:, :2] / (projected[:, 2:3] + 1e-8)
            inside = ((uv[:, 0] >= -0.5) & (uv[:, 0] < width - 0.5)
                      & (uv[:, 1] >= -0.5) & (uv[:, 1] < height - 0.5))
            counts += torch.bincount(self.full_sources[start:end][keep][inside], minlength=self.num_references)
        return counts[self.candidates].cpu().numpy()


class CudaCoverage:
    """Keep large coverage masks on GPU; only small scores/diagnostics go to CPU."""
    def __init__(self, masks):
        self.masks = masks
        self.counts = torch.stack([row.sum(dim=1, dtype=torch.int32) for row in masks])
        self.available = masks.any(dim=1).sum(dim=1, dtype=torch.int32)

    def contribution(self):
        return self.counts.sum(dim=0).cpu().numpy()

    def diagnostic(self, selected):
        covered = torch.zeros((len(self.masks), self.masks.shape[2]), dtype=torch.bool, device='cuda')
        for source in selected:
            covered |= self.masks[:, source]
        available = self.available.cpu().numpy()
        fractions = covered.sum(dim=1).cpu().numpy() / np.maximum(1, available)
        return dict(selected_coverage_fraction_per_target=fractions.tolist(),
                    sampled_visible_pixels_per_target=available.tolist(),
                    mean_coverage=float(fractions.mean()), min_coverage=float(fractions.min()),
                    targets_without_sampled_candidate_coverage=int(np.sum(available == 0)))

    @torch.inference_mode()
    def choose(self, count):
        targets, references, pixels = self.masks.shape
        if not 1 <= count <= references:
            raise ValueError('Reference count must fit reference pool')
        if (self.counts.max(dim=1).values == 0).any().item():
            raise ValueError('A target has no sampled visible points; inspect pose/coverage before generation')
        best_per_target = self.counts.argmax(dim=1).cpu().tolist()
        selected = list(dict.fromkeys(best_per_target)) if count >= targets else []
        covered = torch.zeros((targets, pixels), dtype=torch.bool, device='cuda')
        for source in selected:
            covered |= self.masks[:, source]
        normalization = self.available.clamp_min(1).to(torch.float64)
        # Match NumPy's secondary preference on the same small count matrix.
        total = (self.counts.cpu().numpy() / normalization.cpu().numpy()[:, None]).sum(axis=0)
        while len(selected) < count:
            score = torch.zeros(references, dtype=torch.float64, device='cuda')
            weights = (torch.ones(targets, device='cuda') if count >= targets else
                       1 / (0.05 + covered.sum(dim=1) / normalization))
            for target in range(targets):
                gain = (self.masks[target] & ~covered[target]).sum(dim=1, dtype=torch.int32)
                score += gain / normalization[target] * weights[target]
            values = score.cpu().numpy()
            best = max((i for i in range(references) if i not in selected),
                       key=lambda i: (values[i], total[i], -i))
            selected.append(best)
            covered |= self.masks[:, best]
        diagnostic = self.diagnostic(selected)
        diagnostic.update(selection_order=selected, best_reference_per_target=best_per_target,
                          selected_pixels_per_target=self.counts[:, selected].cpu().tolist())
        return selected, diagnostic
