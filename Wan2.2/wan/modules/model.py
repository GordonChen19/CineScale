# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
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
        self.attn_scale = 1.0
        self.anchor_attn_stride = 1
        self.anchor_attn_scale = 0.0
        self.anchor_attn_local_window = 0
        self.anchor_attn_local_halo = 0
        self.block_tiled_attn_enabled = False
        self.block_tiled_attn_tile_h = 0
        self.block_tiled_attn_tile_w = 0
        self.block_tiled_attn_stride_h = 0
        self.block_tiled_attn_stride_w = 0
        self.block_tiled_attn_halo = 0
        self.block_tiled_attn_global_stride = 0
        self.block_tiled_attn_global_scale = 1.0
        self.block_tiled_attn_routed_topk = 0
        self.block_tiled_attn_routed_grid = 3
        self.block_tiled_attn_global_attention_mode = "separate"
        self.block_tiled_attn_global_rope_threshold = 24.0

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

        q, k, v = qkv_fn(x)

        if self.block_tiled_attn_enabled:
            x = self.block_tiled_self_attention(q, k, v, seq_lens,
                                                grid_sizes, freqs)
        else:
            q = rope_apply(q, grid_sizes, freqs)
            k = rope_apply(k, grid_sizes, freqs)

            if (self.anchor_attn_stride > 1 and self.anchor_attn_scale != 0
                and self.anchor_attn_local_window > 0):
                x = self.anchor_attention(q, k, v, seq_lens, grid_sizes)
            else:
                x = flash_attention(
                    q=q,
                    k=k,
                    v=v,
                    k_lens=seq_lens,
                    window_size=self.window_size,
                    softmax_scale=self.attn_scale / math.sqrt(d))

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def block_tiled_self_attention(self, q, k, v, seq_lens, grid_sizes, freqs):
        out = torch.zeros_like(v)
        scale = self.attn_scale / math.sqrt(self.head_dim)
        tile_h = self.block_tiled_attn_tile_h
        tile_w = self.block_tiled_attn_tile_w
        stride_h = self.block_tiled_attn_stride_h
        stride_w = self.block_tiled_attn_stride_w
        halo = self.block_tiled_attn_halo
        global_stride = self.block_tiled_attn_global_stride
        global_scale = self.block_tiled_attn_global_scale
        routed_topk = self.block_tiled_attn_routed_topk
        routed_grid = self.block_tiled_attn_routed_grid
        global_attention_mode = self.block_tiled_attn_global_attention_mode
        rope_threshold = self.block_tiled_attn_global_rope_threshold
        train_720_h = 45
        train_720_w = 80
        if min(tile_h, tile_w, stride_h, stride_w) <= 0:
            raise ValueError("Block tiled self-attention tile/stride values must be positive.")
        if global_stride < 0:
            raise ValueError("Block tiled self-attention global stride must be non-negative.")
        if global_scale < 0:
            raise ValueError("Block tiled self-attention global scale must be non-negative.")
        if routed_topk < 0:
            raise ValueError("Block tiled self-attention routed top-k must be non-negative.")
        if routed_grid <= 0:
            raise ValueError("Block tiled self-attention routed grid size must be positive.")
        if global_attention_mode not in ("separate", "joint"):
            raise ValueError(
                "Block tiled self-attention global attention mode must be "
                "'separate' or 'joint'.")
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


        def directional_thresholds(center, max_pos, target_center,
                                   target_max_pos):
            total = 2.0 * rope_threshold
            neg_room = max(float(center), 0.0)
            pos_room = max(float(max_pos) - float(center), 0.0)
            target_neg_room = max(float(target_center), 0.0)
            target_pos_room = max(
                float(target_max_pos) - float(target_center), 0.0)

            t_neg = min(rope_threshold, neg_room)
            t_pos = min(rope_threshold, pos_room)
            unused = total - (t_neg + t_pos)

            pos_extra = min(unused, max(pos_room - t_pos, 0.0))
            t_pos += pos_extra
            unused -= pos_extra

            neg_extra = min(unused, max(neg_room - t_neg, 0.0))
            t_neg += neg_extra

            # Keep the uncompressed region itself inside the Wan2.2 720p RoPE
            # coordinate range. This matters near tile/canvas edges.
            t_neg = min(t_neg, target_neg_room)
            t_pos = min(t_pos, target_pos_room)
            return t_neg, t_pos

        def side_compression(full_room, target_room, threshold):
            full_tail = max(float(full_room) - float(threshold), 0.0)
            target_tail = max(float(target_room) - float(threshold), 0.0)
            if full_tail <= target_tail or full_tail <= 0:
                return 1.0
            if target_tail <= 1e-6:
                return 1e6
            return full_tail / target_tail

        def warp_relative_positions(pos, full_center, local_center, max_pos,
                                    train_max_pos):
            delta = pos.float() - float(full_center)
            target_center = min(max(float(local_center), 0.0),
                                float(train_max_pos))
            t_neg, t_pos = directional_thresholds(full_center, max_pos,
                                                  target_center,
                                                  train_max_pos)
            full_neg_room = max(float(full_center), 0.0)
            full_pos_room = max(float(max_pos) - float(full_center), 0.0)
            target_neg_room = max(target_center, 0.0)
            target_pos_room = max(float(train_max_pos) - target_center, 0.0)
            s_neg = side_compression(full_neg_room, target_neg_room, t_neg)
            s_pos = side_compression(full_pos_room, target_pos_room, t_pos)

            t_neg = torch.tensor(t_neg, device=pos.device, dtype=torch.float32)
            t_pos = torch.tensor(t_pos, device=pos.device, dtype=torch.float32)
            s_neg = torch.tensor(s_neg, device=pos.device, dtype=torch.float32)
            s_pos = torch.tensor(s_pos, device=pos.device, dtype=torch.float32)
            warped = torch.where(
                delta < -t_neg,
                -(t_neg + (delta.abs() - t_neg) / s_neg),
                torch.where(delta > t_pos,
                            t_pos + (delta - t_pos) / s_pos, delta))
            return torch.round(target_center +
                               warped).long().clamp(0, train_max_pos)

        def global_rope_indices(frame_idx, y_idx, x_idx, full_center_y,
                                full_center_x, local_center_y,
                                local_center_x):
            return (
                frame_idx,
                warp_relative_positions(y_idx, full_center_y, local_center_y,
                                        h - 1, train_720_h - 1),
                warp_relative_positions(x_idx, full_center_x, local_center_x,
                                        w - 1, train_720_w - 1),
                freqs,
            )

        for batch_idx, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = int(seq_lens[batch_idx].item())
            q_grid = q[batch_idx, :seq_len].view(f, h, w, self.num_heads,
                                                 self.head_dim)
            k_grid = k[batch_idx, :seq_len].view(f, h, w, self.num_heads,
                                                 self.head_dim)
            v_grid = v[batch_idx, :seq_len].view(f, h, w, self.num_heads,
                                                 self.head_dim)
            canvas = torch.zeros_like(v_grid)
            weights = torch.zeros(
                f, h, w, 1, 1, device=v.device, dtype=torch.float32)
            global_k = None
            global_v = None
            global_y = None
            global_x = None
            global_frame = None
            route_k = None
            route_bounds = None
            if global_stride > 0 and global_scale > 0:
                k_sparse = k_grid[:, ::global_stride, ::global_stride]
                v_sparse = v_grid[:, ::global_stride, ::global_stride]
                sparse_len = f * k_sparse.shape[1] * k_sparse.shape[2]
                global_k = k_sparse.reshape(1, sparse_len, self.num_heads,
                                            self.head_dim)
                global_v = v_sparse.reshape(1, sparse_len, self.num_heads,
                                            self.head_dim)
                ys = torch.arange(
                    0, h, global_stride, device=v.device, dtype=torch.long)
                xs = torch.arange(
                    0, w, global_stride, device=v.device, dtype=torch.long)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                global_y = yy.reshape(1, -1).expand(f, -1).reshape(-1)
                global_x = xx.reshape(1, -1).expand(f, -1).reshape(-1)
                global_frame = torch.arange(
                    f, device=v.device, dtype=torch.long).view(f, 1).expand(
                        f, yy.numel()).reshape(-1)
            if routed_topk > 0 and global_scale > 0:
                route_keys = []
                route_bounds_list = []
                for frame in range(f):
                    for gy0 in range(0, h, routed_grid):
                        gy1 = min(gy0 + routed_grid, h)
                        for gx0 in range(0, w, routed_grid):
                            gx1 = min(gx0 + routed_grid, w)
                            k_cell = k_grid[frame, gy0:gy1, gx0:gx1].mean(
                                dim=(0, 1)).unsqueeze(0)
                            route_keys.append(k_cell.squeeze(0))
                            route_bounds_list.append(
                                (frame, gy0, gy1, gx0, gx1))
                route_k = torch.stack(route_keys, dim=0)
                route_bounds = torch.tensor(
                    route_bounds_list,
                    dtype=torch.long,
                    device=v.device)

            for y0 in starts(h, tile_h, stride_h):
                for x0 in starts(w, tile_w, stride_w):
                    y1 = min(y0 + tile_h, h)
                    x1 = min(x0 + tile_w, w)
                    cy0 = max(0, y0 - halo)
                    cy1 = min(h, y1 + halo)
                    cx0 = max(0, x0 - halo)
                    cx1 = min(w, x1 + halo)
                    iy0, iy1 = y0 - cy0, y1 - cy0
                    ix0, ix1 = x0 - cx0, x1 - cx0
                    full_center_y = (y0 + y1 - 1) / 2.0
                    full_center_x = (x0 + x1 - 1) / 2.0
                    local_center_y = (iy0 + iy1 - 1) / 2.0
                    local_center_x = (ix0 + ix1 - 1) / 2.0

                    q_crop = q_grid[:, cy0:cy1, cx0:cx1]
                    k_crop = k_grid[:, cy0:cy1, cx0:cx1]
                    v_crop = v_grid[:, cy0:cy1, cx0:cx1]
                    crop_grid = torch.tensor(
                        [[f, cy1 - cy0, cx1 - cx0]],
                        dtype=torch.long,
                        device=grid_sizes.device)
                    crop_len = f * (cy1 - cy0) * (cx1 - cx0)
                    q_flat = q_crop.reshape(1, crop_len, self.num_heads,
                                            self.head_dim)
                    k_flat = k_crop.reshape(1, crop_len, self.num_heads,
                                            self.head_dim)
                    v_flat = v_crop.reshape(1, crop_len, self.num_heads,
                                            self.head_dim)
                    q_flat = rope_apply(q_flat, crop_grid, freqs)
                    k_flat = rope_apply(k_flat, crop_grid, freqs)
                    global_k_flat = None
                    global_v_flat = None
                    global_q_flat = q_flat
                    joint_q_flat = q_flat
                    joint_k_flat = k_flat
                    if route_k is not None:
                        selected_k = []
                        selected_v = []
                        for frame in range(f):
                            route_queries = []
                            route_query_frames = []
                            route_query_ys = []
                            route_query_xs = []
                            for qy0 in range(y0, y1, routed_grid):
                                qy1 = min(qy0 + routed_grid, y1)
                                for qx0 in range(x0, x1, routed_grid):
                                    qx1 = min(qx0 + routed_grid, x1)
                                    route_queries.append(
                                        q_grid[frame, qy0:qy1,
                                               qx0:qx1].mean(dim=(0, 1)))
                                    route_query_frames.append(frame)
                                    route_query_ys.append((qy0 + qy1 - 1) // 2)
                                    route_query_xs.append((qx0 + qx1 - 1) // 2)

                            route_q = torch.stack(route_queries, dim=0)
                            route_frame_idx = torch.tensor(
                                route_query_frames,
                                device=v.device,
                                dtype=torch.long)
                            route_y_idx = torch.tensor(
                                route_query_ys,
                                device=v.device,
                                dtype=torch.long)
                            route_x_idx = torch.tensor(
                                route_query_xs,
                                device=v.device,
                                dtype=torch.long)
                            route_frame_idx, route_y_idx, route_x_idx, route_freqs = (
                                global_rope_indices(
                                    route_frame_idx, route_y_idx, route_x_idx,
                                    full_center_y, full_center_x,
                                    local_center_y, local_center_x))
                            q_route = rope_apply_absolute(
                                route_q.reshape(route_q.shape[0],
                                                self.num_heads,
                                                self.head_dim),
                                route_frame_idx, route_y_idx, route_x_idx,
                                route_freqs).mean(dim=1).float()

                            same_frame = route_bounds[:, 0] == frame
                            if not bool(same_frame.any().item()):
                                continue
                            frame_indices = same_frame.nonzero(
                                as_tuple=False).flatten()
                            frame_bounds = route_bounds[frame_indices]
                            frame_route_k = route_k[frame_indices]
                            route_frame = frame_bounds[:, 0]
                            route_y = (frame_bounds[:, 1] +
                                       frame_bounds[:, 2] - 1) // 2
                            route_x = (frame_bounds[:, 3] +
                                       frame_bounds[:, 4] - 1) // 2
                            route_frame, route_y, route_x, route_freqs = (
                                global_rope_indices(
                                    route_frame, route_y, route_x,
                                    full_center_y, full_center_x,
                                    local_center_y, local_center_x))
                            k_route = rope_apply_absolute(
                                frame_route_k.reshape(
                                    frame_route_k.shape[0], self.num_heads,
                                    self.head_dim), route_frame, route_y,
                                route_x, route_freqs).mean(dim=1).float()
                            q_route = torch.nn.functional.normalize(
                                q_route, dim=1)
                            k_route = torch.nn.functional.normalize(
                                k_route, dim=1)
                            scores = torch.matmul(k_route,
                                                  q_route.T).max(dim=1).values
                            outside = (
                                (frame_bounds[:, 2] <= cy0) |
                                (frame_bounds[:, 1] >= cy1) |
                                (frame_bounds[:, 4] <= cx0) |
                                (frame_bounds[:, 3] >= cx1))
                            scores = scores.masked_fill(~outside, -torch.inf)
                            valid_count = int(outside.sum().item())
                            if valid_count <= 0:
                                continue
                            topk = min(routed_topk, valid_count)
                            selected = frame_indices[
                                torch.topk(scores, k=topk).indices]
                            for route_index in selected.tolist():
                                frame, gy0, gy1, gx0, gx1 = (
                                    route_bounds[route_index].tolist())
                                k_cell = k_grid[frame:frame + 1, gy0:gy1,
                                                gx0:gx1]
                                k_cell = k_cell.reshape(
                                    (gy1 - gy0) * (gx1 - gx0),
                                    self.num_heads, self.head_dim)
                                cell_frame_idx = torch.full(
                                    (k_cell.size(0),),
                                    frame,
                                    device=v.device,
                                    dtype=torch.long)
                                cell_y_idx = torch.arange(
                                    gy0, gy1, device=v.device,
                                    dtype=torch.long).view(
                                        gy1 - gy0, 1).expand(
                                            gy1 - gy0, gx1 - gx0).reshape(-1)
                                cell_x_idx = torch.arange(
                                    gx0, gx1, device=v.device,
                                    dtype=torch.long).view(
                                        1, gx1 - gx0).expand(
                                            gy1 - gy0, gx1 - gx0).reshape(-1)
                                cell_frame_idx, cell_y_idx, cell_x_idx, cell_freqs = (
                                    global_rope_indices(
                                        cell_frame_idx, cell_y_idx,
                                        cell_x_idx, full_center_y,
                                        full_center_x, local_center_y,
                                        local_center_x))
                                selected_k.append(
                                    rope_apply_absolute(
                                        k_cell, cell_frame_idx, cell_y_idx,
                                        cell_x_idx, cell_freqs))
                                selected_v.append(
                                    v_grid[frame:frame + 1, gy0:gy1,
                                           gx0:gx1].reshape(
                                               -1, self.num_heads,
                                               self.head_dim))
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
                            g_frame, g_y, g_x, full_center_y, full_center_x,
                            local_center_y, local_center_x)
                        global_k_flat = rope_apply_absolute(
                            global_k_flat.reshape(
                                global_k_flat.shape[1], self.num_heads,
                                self.head_dim), g_frame, g_y, g_x,
                            g_freqs).reshape(
                                1, global_k_flat.shape[1], self.num_heads,
                                self.head_dim)
                    if (global_attention_mode == "joint"
                            and global_k_flat is not None
                            and global_v_flat is not None):
                        k_all = torch.cat([joint_k_flat, global_k_flat],
                                          dim=1)
                        v_all = torch.cat(
                            [v_flat, global_scale * global_v_flat], dim=1)
                        y_crop = flash_attention(
                            q=joint_q_flat,
                            k=k_all,
                            v=v_all,
                            softmax_scale=scale).view(
                                f, cy1 - cy0, cx1 - cx0, self.num_heads,
                                self.head_dim)
                    else:
                        y_crop = flash_attention(
                            q=q_flat,
                            k=k_flat,
                            v=v_flat,
                            softmax_scale=scale).view(
                                f, cy1 - cy0, cx1 - cx0, self.num_heads,
                                self.head_dim)
                    if (global_attention_mode == "separate"
                            and global_k_flat is not None
                            and global_v_flat is not None):
                        y_global = flash_attention(
                            q=global_q_flat,
                            k=global_k_flat,
                            v=global_v_flat,
                            softmax_scale=scale).view(
                                f, cy1 - cy0, cx1 - cx0, self.num_heads,
                                self.head_dim)
                        y_crop = y_crop + global_scale * y_global
                    y_inner = y_crop[:, iy0:iy1, ix0:ix1]
                    blend = blend_window(y1 - y0, x1 - x0, v.device,
                                         torch.float32)
                    canvas[:, y0:y1, x0:x1] += (
                        y_inner.float() * blend).to(canvas.dtype)
                    weights[:, y0:y1, x0:x1] += blend

            out[batch_idx, :seq_len] = (
                canvas / weights.clamp_min(1e-8).to(canvas.dtype)).reshape(
                    seq_len, self.num_heads, self.head_dim)

        return out

    def anchor_attention(self, q, k, v, seq_lens, grid_sizes):
        out = torch.zeros_like(v)
        scale = self.attn_scale / math.sqrt(self.head_dim)

        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = int(seq_lens[i].item())
            window = self.anchor_attn_local_window
            halo = self.anchor_attn_local_halo
            stride = self.anchor_attn_stride

            q_i = q[i, :seq_len].view(f, h, w, self.num_heads, self.head_dim)
            k_i = k[i, :seq_len].view(f, h, w, self.num_heads, self.head_dim)
            v_i = v[i, :seq_len].view(f, h, w, self.num_heads, self.head_dim)

            y_local = self.local_window_attention(q_i, k_i, v_i, window,
                                                  halo, scale)

            q_flat = q_i.reshape(1, seq_len, self.num_heads, self.head_dim)
            k_anchor = self.pool_anchor_tokens(k_i, stride)
            v_anchor = self.pool_anchor_tokens(v_i, stride)
            y_anchor = flash_attention(
                q=q_flat,
                k=k_anchor,
                v=v_anchor,
                softmax_scale=scale).view(f, h, w, self.num_heads,
                                          self.head_dim)

            y_i = y_local + self.anchor_attn_scale * y_anchor
            out[i, :seq_len] = y_i.reshape(seq_len, self.num_heads,
                                           self.head_dim)

        return out

    def pool_anchor_tokens(self, x, stride):
        f, h, w, n, d = x.shape
        pad_h = (stride - h % stride) % stride
        pad_w = (stride - w % stride) % stride
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_w, 0, pad_h))
        hp, wp = h + pad_h, w + pad_w
        x = x.view(f, hp // stride, stride, wp // stride, stride, n, d)
        return x.mean(dim=(2, 4)).reshape(1, -1, n, d)

    def local_window_attention(self, q, k, v, window, halo, softmax_scale):
        f, h, w, n, d = q.shape
        pad_h = (window - h % window) % window
        pad_w = (window - w % window) % window

        def pad_query(x):
            if pad_h or pad_w:
                x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_w, 0,
                                                pad_h))
            hp, wp = h + pad_h, w + pad_w
            x = x.view(f, hp // window, window, wp // window, window, n, d)
            x = x.permute(0, 1, 3, 2, 4, 5, 6).reshape(
                -1, window * window, n, d)
            return x, hp, wp

        q_win, hp, wp = pad_query(q)

        if halo <= 0:
            k_win, _, _ = pad_query(k)
            v_win, _, _ = pad_query(v)
        else:
            kv_window = window + 2 * halo
            k_pad = torch.nn.functional.pad(
                k, (0, 0, 0, 0, halo, pad_w + halo, halo, pad_h + halo))
            v_pad = torch.nn.functional.pad(
                v, (0, 0, 0, 0, halo, pad_w + halo, halo, pad_h + halo))
            k_blocks = []
            v_blocks = []
            for frame in range(f):
                for y0 in range(0, hp, window):
                    for x0 in range(0, wp, window):
                        k_blocks.append(
                            k_pad[frame, y0:y0 + kv_window,
                                  x0:x0 + kv_window].reshape(
                                      kv_window * kv_window, n, d))
                        v_blocks.append(
                            v_pad[frame, y0:y0 + kv_window,
                                  x0:x0 + kv_window].reshape(
                                      kv_window * kv_window, n, d))
            k_win = torch.stack(k_blocks, dim=0)
            v_win = torch.stack(v_blocks, dim=0)

        y = flash_attention(
            q=q_win,
            k=k_win,
            v=v_win,
            softmax_scale=softmax_scale)
        y = y.view(f, hp // window, wp // window, window, window, n, d)
        y = y.permute(0, 1, 3, 2, 4, 5, 6).reshape(f, hp, wp, n, d)
        return y[:, :h, :w]


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
