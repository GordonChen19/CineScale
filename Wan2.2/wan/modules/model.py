# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.block_tiled_attn_enabled = False
        self.block_tiled_attn_tile_h = 0
        self.block_tiled_attn_tile_w = 0
        self.block_tiled_attn_stride_h = 0
        self.block_tiled_attn_stride_w = 0
        self.block_tiled_attn_halo = 0
        self.block_tiled_attn_full_global = False
        self.block_tiled_attn_global_stride = 0
        self.block_tiled_attn_routed_topk = 0
        self.block_tiled_attn_routed_grid = 3
        self.block_tiled_attn_global_rope_threshold = 24.0
        self.block_tiled_attn_adaptive_rectified_rope = True

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        if self.block_tiled_attn_enabled:
            x = self.block_tiled_self_attention(
                seq_lens,
                grid_sizes,
                freqs,
                hidden_states=x)
        else:
            q, k, v = qkv_fn(x)
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)
            x = flash_attention(
                q=q,
                k=k,
                v=v,
                k_lens=seq_lens,
                window_size=self.window_size,
                softmax_scale=1.0 / math.sqrt(d))

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def block_tiled_self_attention(self,
                                   seq_lens,
                                   grid_sizes,
                                   freqs,
                                   hidden_states=None):
        project_from_hidden = hidden_states is not None
        if project_from_hidden:
            b, s = hidden_states.shape[:2]
            out = hidden_states.new_zeros(
                b, s, self.num_heads, self.head_dim)
            attn_device = hidden_states.device

        scale = 1.0 / math.sqrt(self.head_dim)
        tile_h = self.block_tiled_attn_tile_h
        tile_w = self.block_tiled_attn_tile_w
        stride_h = self.block_tiled_attn_stride_h
        stride_w = self.block_tiled_attn_stride_w
        halo = self.block_tiled_attn_halo
        full_global_attention = self.block_tiled_attn_full_global
        global_stride = self.block_tiled_attn_global_stride
        routed_topk = self.block_tiled_attn_routed_topk
        routed_grid = self.block_tiled_attn_routed_grid
        rope_threshold = self.block_tiled_attn_global_rope_threshold
        adaptive_rectified_rope = self.block_tiled_attn_adaptive_rectified_rope
        max_relative_y = 44
        max_relative_x = 79
        if min(tile_h, tile_w, stride_h, stride_w) <= 0:
            raise ValueError("Block tiled self-attention tile/stride values must be positive.")
        if halo < 0:
            raise ValueError(
                "Block tiled self-attention halo must be non-negative.")
        if global_stride < 0:
            raise ValueError("Block tiled self-attention global stride must be non-negative.")
        if routed_topk < 0:
            raise ValueError("Block tiled self-attention routed top-k must be non-negative.")
        if routed_grid <= 0:
            raise ValueError("Block tiled self-attention routed grid size must be positive.")
        if rope_threshold < 0:
            raise ValueError(
                "Block tiled self-attention global RoPE threshold must be non-negative."
            )


        def starts(length, tile, stride):
            if tile >= length:
                return [0]
            values = list(range(0, length - tile + 1, stride))
            if values[-1] != length - tile:
                values.append(length - tile)
            return values

        def blend_window(height, width, device, dtype):
            wy = (torch.ones(height, device=device, dtype=dtype)
                  if height <= 1 else torch.hann_window(
                      height, periodic=False, device=device, dtype=dtype))
            wx = (torch.ones(width, device=device, dtype=dtype)
                  if width <= 1 else torch.hann_window(
                      width, periodic=False, device=device, dtype=dtype))
            return (wy[:, None] * wx[None, :]).clamp_min(1e-3).view(
                height, width, 1, 1)

        @torch.amp.autocast('cuda', enabled=False)
        def rope_apply_absolute(x, frame_idx, y_idx, x_idx, freqs):
            """Apply RoPE using absolute full-canvas positions per token."""
            if x.dim() != 3:
                raise ValueError(
                    "rope_apply_absolute expects [tokens, heads, head_dim], "
                    f"got {tuple(x.shape)}")
            n, c = x.size(1), x.size(2) // 2
            freq_t, freq_y, freq_x = freqs.split(
                [c - 2 * (c // 3), c // 3, c // 3], dim=1)
            x_complex = torch.view_as_complex(
                x.to(torch.float64).reshape(x.size(0), n, -1, 2))
            freqs_i = torch.cat([
                freq_t[frame_idx],
                freq_y[y_idx],
                freq_x[x_idx],
            ],
                                dim=1).view(x.size(0), 1, -1)
            x_out = torch.view_as_real(x_complex * freqs_i).flatten(2)
            return x_out.float().type_as(x)

        def rope_apply_absolute_chunked(x, frame_idx, y_idx, x_idx, freqs,
                                        chunk_size=32768):
            """Apply absolute RoPE without a full-sequence float64 temporary."""
            if x.size(0) <= chunk_size:
                return rope_apply_absolute(
                    x, frame_idx, y_idx, x_idx, freqs)
            output = torch.empty_like(x)
            for start in range(0, x.size(0), chunk_size):
                end = min(start + chunk_size, x.size(0))
                output[start:end].copy_(
                    rope_apply_absolute(
                        x[start:end],
                        frame_idx[start:end],
                        y_idx[start:end],
                        x_idx[start:end],
                        freqs))
            return output


        def warp_axis_positions(pos, tile_min, tile_max, max_pos,
                                max_relative):
            """Compress only positions whose distance from a tile exceeds the
            training-relative range.

            The result satisfies |key - query| <= max_relative for every query
            position in [tile_min, tile_max]. Nearby positions retain their
            original full-canvas coordinates.
            """
            tile_span = float(tile_max - tile_min)
            if tile_span > float(max_relative):
                raise ValueError(
                    "A self-attention tile exceeds the configured RoPE "
                    f"relative range ({tile_span:g} > {max_relative}). "
                    "Reduce the tile size.")

            external_budget = float(max_relative) - tile_span
            neg_room = float(tile_min)
            pos_room = float(max_pos - tile_max)
            t_neg = min(float(rope_threshold), neg_room, external_budget)
            t_pos = min(float(rope_threshold), pos_room, external_budget)

            left_keep = float(tile_min) - t_neg
            right_keep = float(tile_max) + t_pos
            left_target = max(0.0, float(tile_max) - float(max_relative))
            right_target = min(float(max_pos),
                               float(tile_min) + float(max_relative))

            pos_float = pos.float()
            if left_keep > 0.0 and left_target > 0.0:
                left_warped = left_target + (
                    pos_float / left_keep) * (left_keep - left_target)
            else:
                left_warped = pos_float

            right_source_span = float(max_pos) - right_keep
            right_target_span = right_target - right_keep
            if right_source_span > 0.0 and right_target < float(max_pos):
                right_warped = right_keep + (
                    (pos_float - right_keep) / right_source_span
                ) * right_target_span
            else:
                right_warped = pos_float

            warped = torch.where(
                pos_float < left_keep,
                left_warped,
                torch.where(pos_float > right_keep, right_warped, pos_float))
            return torch.round(warped).long().clamp(0, max_pos)

        def global_rope_indices(frame_idx, y_idx, x_idx, tile_y_min,
                                tile_y_max, tile_x_min, tile_x_max):
            if not adaptive_rectified_rope:
                return (
                    frame_idx.clamp(0, freqs.shape[0] - 1),
                    y_idx.clamp(0, freqs.shape[0] - 1),
                    x_idx.clamp(0, freqs.shape[0] - 1),
                    freqs,
                )
            return (
                frame_idx,
                warp_axis_positions(y_idx, tile_y_min, tile_y_max, h - 1,
                                    max_relative_y),
                warp_axis_positions(x_idx, tile_x_min, tile_x_max, w - 1,
                                    max_relative_x),
                freqs,
            )

        tile_world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized() else 1)
        tile_rank = (
            dist.get_rank()
            if dist.is_available() and dist.is_initialized() else 0)

        for batch_idx, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = int(seq_lens[batch_idx].item())
            if project_from_hidden:
                hidden_grid = hidden_states[batch_idx, :seq_len].view(
                    f, h, w, self.dim)
            canvas = out[batch_idx, :seq_len].view(
                f, h, w, self.num_heads, self.head_dim)
            weights = torch.zeros(
                f, h, w, 1, 1, device=attn_device, dtype=torch.float32)
            global_k = None
            global_v = None
            global_y = None
            global_x = None
            global_frame = None
            route_k = None
            route_bounds = None
            all_k_flat = None
            all_v_flat = None
            all_frame_idx = None
            all_y_idx = None
            all_x_idx = None
            if full_global_attention:
                if torch.is_grad_enabled():
                    raise RuntimeError(
                        "Full-global tiled self-attention is an inference-only "
                        "path and must run with gradients disabled.")
                hidden_flat = hidden_grid.reshape(seq_len, self.dim)
                projection_chunk_size = 8192
                first_end = min(projection_chunk_size, seq_len)
                first_hidden = hidden_flat[:first_end]
                first_k = self.norm_k(self.k(first_hidden)).view(
                    first_end, self.num_heads, self.head_dim)
                first_v = self.v(first_hidden).view(
                    first_end, self.num_heads, self.head_dim)
                all_k_flat = first_k.new_empty(
                    seq_len, self.num_heads, self.head_dim)
                all_v_flat = first_v.new_empty(
                    seq_len, self.num_heads, self.head_dim)
                all_k_flat[:first_end].copy_(first_k)
                all_v_flat[:first_end].copy_(first_v)
                del first_hidden, first_k, first_v
                for start in range(first_end, seq_len,
                                   projection_chunk_size):
                    end = min(start + projection_chunk_size, seq_len)
                    hidden_chunk = hidden_flat[start:end]
                    all_k_flat[start:end].copy_(
                        self.norm_k(self.k(hidden_chunk)).view(
                            end - start, self.num_heads, self.head_dim))
                    all_v_flat[start:end].copy_(
                        self.v(hidden_chunk).view(
                            end - start, self.num_heads, self.head_dim))
                all_frame_idx = torch.arange(
                    f, device=attn_device, dtype=torch.long).view(
                        f, 1, 1).expand(f, h, w).reshape(-1)
                all_y_idx = torch.arange(
                    h, device=attn_device, dtype=torch.long).view(
                        1, h, 1).expand(f, h, w).reshape(-1)
                all_x_idx = torch.arange(
                    w, device=attn_device, dtype=torch.long).view(
                        1, 1, w).expand(f, h, w).reshape(-1)
            elif global_stride > 0:
                if project_from_hidden:
                    hidden_sparse = hidden_grid[:, ::global_stride,
                                                ::global_stride]
                    sparse_shape = hidden_sparse.shape[:3]
                    hidden_sparse = hidden_sparse.reshape(-1, self.dim)
                    k_sparse = self.norm_k(self.k(hidden_sparse)).view(
                        *sparse_shape, self.num_heads, self.head_dim)
                    v_sparse = self.v(hidden_sparse).view(
                        *sparse_shape, self.num_heads, self.head_dim)
        
                sparse_len = f * k_sparse.shape[1] * k_sparse.shape[2]
                global_k = k_sparse.reshape(1, sparse_len, self.num_heads,
                                            self.head_dim)
                global_v = v_sparse.reshape(1, sparse_len, self.num_heads,
                                            self.head_dim)
                ys = torch.arange(
                    0, h, global_stride, device=attn_device, dtype=torch.long)
                xs = torch.arange(
                    0, w, global_stride, device=attn_device, dtype=torch.long)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                global_y = yy.reshape(1, -1).expand(f, -1).reshape(-1)
                global_x = xx.reshape(1, -1).expand(f, -1).reshape(-1)
                global_frame = torch.arange(
                    f, device=attn_device, dtype=torch.long).view(f, 1).expand(
                        f, yy.numel()).reshape(-1)
            if routed_topk > 0 and not full_global_attention:
                route_keys = []
                route_bounds_list = []
                for frame in range(f):
                    for gy0 in range(0, h, routed_grid):
                        gy1 = min(gy0 + routed_grid, h)
                        for gx0 in range(0, w, routed_grid):
                            gx1 = min(gx0 + routed_grid, w)
                            if project_from_hidden:
                                cell_hidden = hidden_grid[
                                    frame, gy0:gy1, gx0:gx1].reshape(
                                        -1, self.dim)
                                k_cell = self.norm_k(
                                    self.k(cell_hidden)).view(
                                        -1, self.num_heads,
                                        self.head_dim).mean(
                                            dim=0, keepdim=True)
                            
                            route_keys.append(k_cell.squeeze(0))
                            route_bounds_list.append(
                                (frame, gy0, gy1, gx0, gx1))
                route_k = torch.stack(route_keys, dim=0)
                route_bounds = torch.tensor(
                    route_bounds_list,
                    dtype=torch.long,
                    device=attn_device)

            tile_coords = [
                (y0, x0)
                for y0 in starts(h, tile_h, stride_h)
                for x0 in starts(w, tile_w, stride_w)
            ]
            if tile_coords:
                for tile_index, (y0, x0) in enumerate(tile_coords):
                    if tile_index % tile_world_size != tile_rank:
                        continue
                    y1 = min(y0 + tile_h, h)
                    x1 = min(x0 + tile_w, w)
                    cy0 = max(0, y0 - halo)
                    cy1 = min(h, y1 + halo)
                    cx0 = max(0, x0 - halo)
                    cx1 = min(w, x1 + halo)
                    iy0, iy1 = y0 - cy0, y1 - cy0
                    ix0, ix1 = x0 - cx0, x1 - cx0
                    # Rectified global RoPE is bounded against the output core.
                    # Halo queries are discarded after attention and therefore
                    # must not enlarge the core's positional budget.
                    tile_y_min, tile_y_max = y0, y1 - 1
                    tile_x_min, tile_x_max = x0, x1 - 1

                    crop_grid = torch.tensor(
                        [[f, cy1 - cy0, cx1 - cx0]],
                        dtype=torch.long,
                        device=grid_sizes.device)
                    crop_len = f * (cy1 - cy0) * (cx1 - cx0)
                    if project_from_hidden:
                        hidden_crop = hidden_grid[:, cy0:cy1,
                                                  cx0:cx1].reshape(
                                                      1, crop_len, self.dim)
                        q_flat = self.norm_q(self.q(hidden_crop)).view(
                            1, crop_len, self.num_heads, self.head_dim)
                        if not full_global_attention:
                            k_flat = self.norm_k(self.k(hidden_crop)).view(
                                1, crop_len, self.num_heads, self.head_dim)
                            v_flat = self.v(hidden_crop).view(
                                1, crop_len, self.num_heads, self.head_dim)
                            q_crop_grid = q_flat.view(
                                f, cy1 - cy0, cx1 - cx0, self.num_heads,
                                self.head_dim)
                   
                    if adaptive_rectified_rope or full_global_attention:
                        local_frame = torch.arange(
                            f, device=attn_device, dtype=torch.long).view(
                                f, 1, 1).expand(
                                    f, cy1 - cy0, cx1 - cx0).reshape(-1)
                        local_y = torch.arange(
                            cy0, cy1, device=attn_device, dtype=torch.long)
                        local_x = torch.arange(
                            cx0, cx1, device=attn_device, dtype=torch.long)
                        local_y = local_y.view(
                            1, cy1 - cy0, 1).expand(
                                f, cy1 - cy0, cx1 - cx0).reshape(-1)
                        local_x = local_x.view(
                            1, 1, cx1 - cx0).expand(
                                f, cy1 - cy0, cx1 - cx0).reshape(-1)
                        q_flat = rope_apply_absolute(
                            q_flat.squeeze(0), local_frame, local_y, local_x,
                            freqs).unsqueeze(0)
                        if not full_global_attention:
                            k_flat = rope_apply_absolute(
                                k_flat.squeeze(0), local_frame, local_y,
                                local_x, freqs).unsqueeze(0)
                    else:
                        q_flat = rope_apply(q_flat, crop_grid, freqs)
                        if not full_global_attention:
                            k_flat = rope_apply(k_flat, crop_grid, freqs)
                    global_k_flat = None
                    global_v_flat = None
                    if full_global_attention:
                        g_frame, g_y, g_x, g_freqs = global_rope_indices(
                            all_frame_idx, all_y_idx, all_x_idx,
                            tile_y_min, tile_y_max, tile_x_min, tile_x_max)
                        global_k_flat = rope_apply_absolute_chunked(
                            all_k_flat, g_frame, g_y, g_x,
                            g_freqs).unsqueeze(0)
                        y_crop = flash_attention(
                            q=q_flat,
                            k=global_k_flat,
                            v=all_v_flat.unsqueeze(0),
                            softmax_scale=scale).view(
                                f, cy1 - cy0, cx1 - cx0, self.num_heads,
                                self.head_dim)
                        del global_k_flat
                    elif route_k is not None:
                        selected_k = []
                        selected_v = []
                        selected_route_indices = []
                        selected_route_seen = set()
                        cells_per_frame = route_k.shape[0] // f
                        route_k_by_frame = route_k.view(
                            f, cells_per_frame, self.num_heads,
                            self.head_dim)
                        route_bounds_by_frame = route_bounds.view(
                            f, cells_per_frame, 5)
                        query_cells = []
                        query_y_centers = []
                        query_x_centers = []
                        for qy0 in range(y0, y1, routed_grid):
                            qy1 = min(qy0 + routed_grid, y1)
                            for qx0 in range(x0, x1, routed_grid):
                                qx1 = min(qx0 + routed_grid, x1)
                                query_cells.append((qy0, qy1, qx0, qx1))
                                query_y_centers.append((qy0 + qy1 - 1) // 2)
                                query_x_centers.append((qx0 + qx1 - 1) // 2)
                        query_y_centers = torch.tensor(
                            query_y_centers,
                            device=attn_device,
                            dtype=torch.long)
                        query_x_centers = torch.tensor(
                            query_x_centers,
                            device=attn_device,
                            dtype=torch.long)
                        frame_bounds = route_bounds_by_frame
                        frame_route_k = route_k_by_frame
                        route_frame = frame_bounds[:, :, 0].reshape(-1)
                        route_y = (frame_bounds[:, :, 1] +
                                   frame_bounds[:, :, 2] - 1).reshape(
                                       -1) // 2
                        route_x = (frame_bounds[:, :, 3] +
                                   frame_bounds[:, :, 4] - 1).reshape(
                                       -1) // 2
                        route_frame, route_y, route_x, route_freqs = (
                            global_rope_indices(route_frame, route_y, route_x,
                                                tile_y_min, tile_y_max,
                                                tile_x_min, tile_x_max))
                        k_route = rope_apply_absolute(
                            frame_route_k.reshape(f * cells_per_frame,
                                                  self.num_heads,
                                                  self.head_dim),
                            route_frame, route_y, route_x,
                            route_freqs).mean(dim=1).float()
                        k_route = k_route.view(f, cells_per_frame, -1)
                        k_route_flat_all = k_route.reshape(
                            1, f * cells_per_frame, -1)
                        outside = (
                            (frame_bounds[:, :, 2] <= cy0) |
                            (frame_bounds[:, :, 1] >= cy1) |
                            (frame_bounds[:, :, 4] <= cx0) |
                            (frame_bounds[:, :, 3] >= cx1))
                        valid_counts = outside.sum(dim=1)
                        # Batch routing score matmuls over a small frame chunk.
                        # Full-frame batching can allocate very large RoPE
                        # temporaries at 4K, so keep this conservative.
                        route_frame_batch = 2
                        for frame_start in range(0, f, route_frame_batch):
                            frame_end = min(f, frame_start + route_frame_batch)
                            frames_in_chunk = frame_end - frame_start
                            route_queries = [
                                q_crop_grid[
                                    frame_start:frame_end,
                                    qy0 - cy0:qy1 - cy0,
                                    qx0 - cx0:qx1 - cx0].mean(dim=(1, 2))
                                for qy0, qy1, qx0, qx1 in query_cells
                            ]
                            route_q = torch.stack(route_queries, dim=1)
                            query_count = route_q.shape[1]
                            route_q = route_q.reshape(
                                frames_in_chunk * query_count,
                                self.num_heads, self.head_dim)
                            route_frame_idx = torch.arange(
                                frame_start,
                                frame_end,
                                device=attn_device,
                                dtype=torch.long).view(
                                    frames_in_chunk, 1).expand(
                                        frames_in_chunk,
                                        query_count).reshape(-1)
                            route_y_idx = query_y_centers.view(
                                1, query_count).expand(
                                    frames_in_chunk, query_count).reshape(-1)
                            route_x_idx = query_x_centers.view(
                                1, query_count).expand(
                                    frames_in_chunk, query_count).reshape(-1)
                            q_route = rope_apply_absolute(
                                route_q, route_frame_idx, route_y_idx,
                                route_x_idx, freqs).mean(dim=1).float()
                            q_route = q_route.view(frames_in_chunk,
                                                   query_count, -1)

                            k_route_flat = k_route_flat_all.expand(
                                frames_in_chunk, -1, -1)
                            scores = torch.bmm(
                                k_route_flat,
                                q_route.transpose(1, 2)).max(dim=2).values
                            scores = scores.view(frames_in_chunk, f,
                                                 cells_per_frame)
                            scores = scores.masked_fill(
                                ~outside.unsqueeze(0), -torch.inf)
                            for chunk_frame in range(frames_in_chunk):
                                for candidate_frame in range(f):
                                    valid_count = int(valid_counts[
                                        candidate_frame].item())
                                    if valid_count <= 0:
                                        continue
                                    topk = min(routed_topk, valid_count)
                                    selected_local = torch.topk(
                                        scores[chunk_frame, candidate_frame],
                                        k=topk).indices
                                    selected = (
                                        candidate_frame * cells_per_frame +
                                        selected_local)
                                    for route_index in selected.tolist():
                                        if route_index in selected_route_seen:
                                            continue
                                        selected_route_seen.add(route_index)
                                        selected_route_indices.append(route_index)
                        for route_index in selected_route_indices:
                            frame, gy0, gy1, gx0, gx1 = (
                                route_bounds[route_index].tolist())
                            if project_from_hidden:
                                cell_hidden = hidden_grid[
                                    frame, gy0:gy1, gx0:gx1].reshape(
                                        -1, self.dim)
                                k_cell = self.norm_k(
                                    self.k(cell_hidden)).view(
                                        -1, self.num_heads, self.head_dim)
                                v_cell = self.v(cell_hidden).view(
                                    -1, self.num_heads, self.head_dim)
                            
                            cell_frame_idx = torch.full(
                                (k_cell.size(0),),
                                frame,
                                device=attn_device,
                                dtype=torch.long)
                            cell_y_idx = torch.arange(
                                gy0, gy1, device=attn_device,
                                dtype=torch.long).view(
                                    gy1 - gy0, 1).expand(
                                        gy1 - gy0, gx1 - gx0).reshape(-1)
                            cell_x_idx = torch.arange(
                                gx0, gx1, device=attn_device,
                                dtype=torch.long).view(
                                    1, gx1 - gx0).expand(
                                        gy1 - gy0, gx1 - gx0).reshape(-1)
                            cell_frame_idx, cell_y_idx, cell_x_idx, cell_freqs = (
                                global_rope_indices(
                                    cell_frame_idx, cell_y_idx, cell_x_idx,
                                    tile_y_min, tile_y_max, tile_x_min,
                                    tile_x_max))
                            selected_k.append(
                                rope_apply_absolute(
                                    k_cell, cell_frame_idx, cell_y_idx,
                                    cell_x_idx, cell_freqs))
                            selected_v.append(v_cell)
                        if selected_k:
                            selected_k = torch.cat(selected_k, dim=0)
                            selected_v = torch.cat(selected_v, dim=0)
                            global_k_flat = selected_k.reshape(
                                1, selected_k.shape[0], self.num_heads,
                                self.head_dim)
                            global_v_flat = selected_v.reshape(
                                1, selected_v.shape[0], self.num_heads,
                                self.head_dim)
                    elif global_k is not None:
                        global_mask = (
                            (global_y < cy0) | (global_y >= cy1) |
                            (global_x < cx0) | (global_x >= cx1))
                        global_k_flat = global_k[:, global_mask]
                        global_v_flat = global_v[:, global_mask]
                        g_frame = global_frame[global_mask]
                        g_y = global_y[global_mask]
                        g_x = global_x[global_mask]
                        g_frame, g_y, g_x, g_freqs = global_rope_indices(
                            g_frame, g_y, g_x, tile_y_min, tile_y_max,
                            tile_x_min, tile_x_max)
                        global_k_flat = rope_apply_absolute(
                            global_k_flat.reshape(
                                global_k_flat.shape[1], self.num_heads,
                                self.head_dim), g_frame, g_y, g_x,
                            g_freqs).reshape(
                                1, global_k_flat.shape[1], self.num_heads,
                                self.head_dim)
                    if (not full_global_attention and
                            global_k_flat is not None and
                            global_v_flat is not None):
                        k_all = torch.cat([k_flat, global_k_flat], dim=1)
                        v_all = torch.cat([v_flat, global_v_flat], dim=1)
                        y_crop = flash_attention(
                            q=q_flat,
                            k=k_all,
                            v=v_all,
                            softmax_scale=scale).view(
                                f, cy1 - cy0, cx1 - cx0, self.num_heads,
                                self.head_dim)
                    elif not full_global_attention:
                        y_crop = flash_attention(
                            q=q_flat,
                            k=k_flat,
                            v=v_flat,
                            softmax_scale=scale).view(
                                f, cy1 - cy0, cx1 - cx0, self.num_heads,
                                self.head_dim)

                    y_inner = y_crop[:, iy0:iy1, ix0:ix1]
                    blend = blend_window(y1 - y0, x1 - x0, attn_device,
                                         torch.float32)
                    canvas[:, y0:y1, x0:x1] += (
                        y_inner.float() * blend).to(canvas.dtype)
                    weights[:, y0:y1, x0:x1] += blend

            if tile_world_size > 1:
                dist.all_reduce(canvas, op=dist.ReduceOp.SUM)
                dist.all_reduce(weights, op=dist.ReduceOp.SUM)

            canvas.div_(weights.clamp_min(1e-8).to(canvas.dtype))

        return out

class WanCrossAttention(WanSelfAttention):

    def forward(self,
                x,
                context,
                context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(
                self.norm3(x),
                context,
                context_lens
            )
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with oriaginal input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens
)

        for block in self.blocks:
            x = block(x, **kwargs)

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
