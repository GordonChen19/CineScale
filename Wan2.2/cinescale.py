import argparse
import copy
import gc
import logging
import math
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
WAN_ROOT = ROOT if (ROOT / "wan").exists() else ROOT / "Wan2.2"
if str(WAN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_ROOT))



def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "y", "1"):
        return True
    if value in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_torch_dtype(value):
    value = value.lower()
    if value in ("fp32", "float32"):
        return torch.float32
    if value in ("fp16", "float16", "half"):
        return torch.float16
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise argparse.ArgumentTypeError(
        "Expected one of: fp32, fp16, bf16.")


def setup_distributed(args):
    rank = int(os.getenv("RANK", args.rank))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", args.device_id))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size)
        if args.ulysses_size > 1:
            if args.ulysses_size != world_size:
                raise ValueError("--ulysses_size must equal WORLD_SIZE.")
            from wan.distributed.util import init_distributed_group

            init_distributed_group()
    else:
        if args.t5_fsdp or args.dit_fsdp:
            raise ValueError(
                "--t5_fsdp and --dit_fsdp require torchrun with multiple processes."
            )
        if args.ulysses_size > 1:
            raise ValueError(
                "--ulysses_size > 1 requires torchrun with multiple processes.")

    args.rank = rank
    args.device_id = local_rank
    return rank, world_size, local_rank


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def unique_dit_models(model):
    seen = set()
    result = []
    for name in ("low_noise_model", "high_noise_model"):
        dit_model = getattr(model, name, None)
        if dit_model is None or id(dit_model) in seen:
            continue
        seen.add(id(dit_model))
        result.append((name, dit_model))
    return result


def offload_dit_models(model):
    for _, dit_model in unique_dit_models(model):
        if next(dit_model.parameters()).device.type == "cuda":
            dit_model.to("cpu")
    torch.cuda.empty_cache()


def set_vae_dtype(vae, dtype):
    vae.dtype = dtype
    vae.mean = vae.mean.to(dtype=dtype, device=vae.device)
    vae.std = vae.std.to(dtype=dtype, device=vae.device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    vae.model.to(device=vae.device, dtype=dtype)


def offload_vae_model(model):
    vae_model = getattr(getattr(model, "vae", None), "model", None)
    if vae_model is not None and next(vae_model.parameters()).device.type == "cuda":
        vae_model.cpu()
        gc.collect()
        torch.cuda.empty_cache()


def onload_dit_models(model):
    for _, dit_model in unique_dit_models(model):
        if next(dit_model.parameters()).device.type == "cpu":
            dit_model.to(model.device)
    torch.cuda.empty_cache()


def make_model_config(wan_configs):
    cfg = copy.deepcopy(wan_configs["t2v-A14B"])
    return cfg

def parse_size(size):
    width, height = size.lower().split("*")
    return int(width), int(height)


def best_latent_size(width, height, max_area, vae_stride, patch_size):
    aspect_ratio = height / width
    lat_h = round(
        np.sqrt(max_area * aspect_ratio) // vae_stride[1] //
        patch_size[1] * patch_size[1])
    lat_w = round(
        np.sqrt(max_area / aspect_ratio) // vae_stride[2] //
        patch_size[2] * patch_size[2])
    return lat_h, lat_w


def read_video(path, frame_num):
    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Reading videos requires imageio. Install Wan2.2 requirements in "
            "the active environment, e.g. `pip install -r Wan2.2/requirements.txt`."
        ) from exc

    frames = []
    for frame in iio.imiter(path):
        frames.append(Image.fromarray(frame[..., :3]).convert("RGB"))
        if len(frames) == frame_num:
            break
    original_frame_count = len(frames)
    if not frames:
        raise ValueError(f"No frames could be read from {path}")
    while len(frames) < frame_num:
        frames.append(frames[-1].copy())
    return frames, original_frame_count



def resize_frames(frames, height, width):
    resized = []
    for frame in frames:
        frame = frame.resize((width, height), Image.LANCZOS)
        resized.append(TF.to_tensor(frame).sub_(0.5).div_(0.5))
    return torch.stack(resized, dim=1)


def _tile_starts(length, tile, stride):
    if tile >= length:
        return [0]
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return sorted(set(values))


def vae_encode_video_tiled(vae,
                           video,
                           tile_h=1024,
                           tile_w=1024,
                           overlap=128,
                           temporal_pad_frames=4,
                           temporal_stride=4):
    if overlap >= min(tile_h, tile_w):
        raise ValueError("VAE encode tile overlap must be smaller than tile size.")
    if temporal_pad_frames < 0:
        raise ValueError("VAE encode temporal padding must be non-negative.")

    original_video = video
    _, frames, height, width = video.shape
    latent_temporal_crop = int(
        math.ceil(temporal_pad_frames / temporal_stride)
    ) if temporal_pad_frames > 0 else 0
    expected_t = (frames - 1) // temporal_stride + 1
    if temporal_pad_frames > 0:
        temporal_pad = video[:, :1].repeat(1, temporal_pad_frames, 1, 1)
        video = torch.cat([temporal_pad, video], dim=1)

    spatial_stride_h = 8
    spatial_stride_w = 8
    pad_h = int(math.ceil(overlap / spatial_stride_h) * spatial_stride_h)
    pad_w = int(math.ceil(overlap / spatial_stride_w) * spatial_stride_w)
    latent_crop_h = pad_h // spatial_stride_h
    latent_crop_w = pad_w // spatial_stride_w

    if pad_h > 0 or pad_w > 0:
        video = F.pad(
            video,
            (pad_w, pad_w, pad_h, pad_h),
            mode="reflect")

    padded_height = video.shape[-2]
    padded_width = video.shape[-1]
    stride_h = tile_h - overlap
    stride_w = tile_w - overlap
    ys = _tile_starts(padded_height, tile_h, stride_h)
    xs = _tile_starts(padded_width, tile_w, stride_w)
    total_tiles = len(ys) * len(xs)
    pbar = tqdm(
        total=total_tiles,
        desc="VAE Encode",
        unit="tile",
        disable=not is_main_process())

    out_canvas = None
    weight_canvas = None
    scale_h = None
    scale_w = None
    device = vae.device

    try:
        with torch.no_grad():
            for y0 in ys:
                for x0 in xs:
                    y1 = min(y0 + tile_h, padded_height)
                    x1 = min(x0 + tile_w, padded_width)
                    tile = video[:, :, y0:y1, x0:x1].to(device)
                    tile_latent = vae.encode([tile])[0].float()

                    _, latent_t, latent_h, latent_w = tile_latent.shape
                    if out_canvas is None:
                        scale_h = latent_h / (y1 - y0)
                        scale_w = latent_w / (x1 - x0)
                        full_h = round(padded_height * scale_h)
                        full_w = round(padded_width * scale_w)
                        out_canvas = torch.zeros(
                            tile_latent.shape[0],
                            latent_t,
                            full_h,
                            full_w,
                            device=device,
                            dtype=torch.float32)
                        weight_canvas = torch.zeros_like(out_canvas)

                    out_y0 = round(y0 * scale_h)
                    out_x0 = round(x0 * scale_w)
                    out_y1 = out_y0 + latent_h
                    out_x1 = out_x0 + latent_w

                    weight_y = torch.hann_window(
                        latent_h, periodic=False, device=device).clamp_min(
                            1e-3)
                    weight_x = torch.hann_window(
                        latent_w, periodic=False, device=device).clamp_min(
                            1e-3)
                    weight = (weight_y[:, None] * weight_x[None, :]).view(
                        1, 1, latent_h, latent_w)

                    out_canvas[:, :, out_y0:out_y1,
                               out_x0:out_x1] += tile_latent * weight
                    weight_canvas[:, :, out_y0:out_y1,
                                  out_x0:out_x1] += weight

                    del tile, tile_latent, weight
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    pbar.update(1)
    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if tile_h <= 256 or tile_w <= 256:
            raise
        logging.warning(
            "Tiled VAE encode OOM at %dx%d pixel tiles. Retrying with %dx%d tiles.",
            tile_h, tile_w, max(256, tile_h // 2), max(256, tile_w // 2))
        return vae_encode_video_tiled(
            vae,
            original_video,
            tile_h=max(256, tile_h // 2),
            tile_w=max(256, tile_w // 2),
            overlap=min(64, max(0, max(256, tile_h // 2) - 1),
                        max(0, max(256, tile_w // 2) - 1)),
            temporal_pad_frames=temporal_pad_frames,
            temporal_stride=temporal_stride)
    finally:
        pbar.close()

    latent = (out_canvas / weight_canvas.clamp_min(1e-6)).float()
    if latent_temporal_crop > 0:
        latent = latent[:, latent_temporal_crop:]
    if latent_crop_h > 0:
        latent = latent[:, :, latent_crop_h:-latent_crop_h, :]
    if latent_crop_w > 0:
        latent = latent[:, :, :, latent_crop_w:-latent_crop_w]
    expected_h = height // spatial_stride_h
    expected_w = width // spatial_stride_w
    return latent[:, :expected_t, :expected_h, :expected_w].contiguous()


def prepare_text_context(model, prompt, negative_prompt, offload_model):
    if negative_prompt == "":
        negative_prompt = model.sample_neg_prompt

    if not model.t5_cpu:
        model.text_encoder.model.to(model.device)
        context = model.text_encoder([prompt], model.device)
        context_null = model.text_encoder([negative_prompt], model.device)
        if offload_model:
            model.text_encoder.model.cpu()
    else:
        context = model.text_encoder([prompt], torch.device("cpu"))
        context_null = model.text_encoder([negative_prompt], torch.device("cpu"))
        context = [u.to(model.device) for u in context]
        context_null = [u.to(model.device) for u in context_null]

    return context, context_null


def make_scheduler(model, sample_steps, shift):
    from wan.utils.fm_solvers import (
        FlowDPMSolverMultistepScheduler,
        get_sampling_sigmas,
        retrieve_timesteps,
    )

    scheduler = FlowDPMSolverMultistepScheduler(
        num_train_timesteps=model.num_train_timesteps,
        solver_order=1,
        shift=1,
        use_dynamic_shifting=False)
    sampling_sigmas = get_sampling_sigmas(sample_steps, shift)
    timesteps, _ = retrieve_timesteps(
        scheduler, device=model.device, sigmas=sampling_sigmas)
    return timesteps, scheduler.sigmas.to(model.device)


def flow_euler_step(latent, flow, sigma_from, sigma_to):
    return latent + (sigma_to - sigma_from) * flow


def set_self_attention_scale(model, scale):
    if scale <= 0:
        raise ValueError("--self_attn_scale must be positive.")

    def set_model_scale(wan_model):
        target_model = getattr(wan_model, "module", wan_model)
        for block in target_model.blocks:
            block.self_attn.attn_scale = scale

    for _, dit_model in unique_dit_models(model):
        set_model_scale(dit_model)


def parse_block_range(block_range):
    if block_range is None or block_range == "":
        return None
    blocks = set()
    for part in block_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            start = int(start)
            end = int(end)
            if end < start:
                raise ValueError(f"Invalid block range: {part}")
            blocks.update(range(start, end + 1))
        else:
            blocks.add(int(part))
    return blocks

def set_block_tiled_self_attention(model, enabled, tile_height, tile_width,
                                   stride_height, stride_width, halo,
                                   routed_topk, routed_grid,
                                   global_attention_mode,
                                   global_rope_threshold,
                                   block_range):
    if enabled and min(tile_height, tile_width, stride_height, stride_width) <= 0:
        raise ValueError(
            "Block tiled self-attention tile and stride dimensions must be positive."
        )
    if enabled and halo < 0:
        raise ValueError("--block_tiled_self_attn_halo must be non-negative.")
    if enabled and routed_topk < 0:
        raise ValueError(
            "--block_tiled_self_attn_routed_topk must be non-negative.")
    if enabled and routed_grid <= 0:
        raise ValueError(
            "--block_tiled_self_attn_routed_grid must be positive.")
    if enabled and global_attention_mode not in ("separate", "joint"):
        raise ValueError(
            "--block_tiled_self_attn_global_attention_mode must be 'separate' or 'joint'."
        )
    if enabled and global_rope_threshold < 0:
        raise ValueError(
            "--block_tiled_self_attn_global_rope_threshold must be non-negative."
        )
    selected_blocks = parse_block_range(block_range)

    def set_model_block_tiling(wan_model):
        target_model = getattr(wan_model, "module", wan_model)
        for i, block in enumerate(target_model.blocks):
            enabled_block = enabled and (selected_blocks is None or i in selected_blocks)
            block.self_attn.block_tiled_attn_enabled = enabled_block
            block.self_attn.block_tiled_attn_tile_h = tile_height
            block.self_attn.block_tiled_attn_tile_w = tile_width
            block.self_attn.block_tiled_attn_stride_h = stride_height
            block.self_attn.block_tiled_attn_stride_w = stride_width
            block.self_attn.block_tiled_attn_halo = halo
            block.self_attn.block_tiled_attn_routed_topk = (
                routed_topk if enabled_block else 0)
            block.self_attn.block_tiled_attn_routed_grid = routed_grid
            block.self_attn.block_tiled_attn_global_attention_mode = (
                global_attention_mode)
            block.self_attn.block_tiled_attn_global_rope_threshold = (
                global_rope_threshold)

    for _, dit_model in unique_dit_models(model):
        set_model_block_tiling(dit_model)


def predict_cond_uncond(model,
                        latent,
                        timestep,
                        context,
                        context_null,
                        seq_len,
                        guide_scale,
                        boundary,
                        offload_model,
                        y=None):
    timestep = torch.stack([timestep]).to(model.device)
    active_model = model._prepare_model_for_timestep(timestep[0], boundary,
                                                     offload_model)
    step_guide_scale = guide_scale[1] if timestep[0].item(
    ) >= boundary else guide_scale[0]

    arg_c = {"context": [context[0]], "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    if y is not None:
        arg_c["y"] = [y]
        arg_null["y"] = [y]
    latent_input = latent.to(model.device)
    latent_model_input = [latent_input]

    with torch.no_grad():
        flow_cond = active_model(latent_model_input, t=timestep, **arg_c)[0]
        flow_uncond = active_model(latent_model_input, t=timestep,
                                   **arg_null)[0]

    if offload_model:
        torch.cuda.empty_cache()

    flow = flow_uncond + step_guide_scale * (flow_cond - flow_uncond)
    return flow, flow_cond, flow_uncond

def compute_seq_len(model, latent_shape):
    _, latent_frames, lat_h, lat_w = latent_shape
    seq_len = latent_frames * lat_h * lat_w // (
        model.patch_size[1] * model.patch_size[2])
    return int(math.ceil(seq_len / model.sp_size)) * model.sp_size


def encode_video_inputs(model, video_path, frame_num, size=None):
    frames, input_frame_count = read_video(video_path, frame_num)
    if size is None:
        max_area = frames[0].width * frames[0].height
    else:
        requested_width, requested_height = parse_size(size)
        max_area = requested_width * requested_height
    lat_h, lat_w = best_latent_size(frames[0].width, frames[0].height,
                                    max_area, model.vae_stride,
                                    model.patch_size)
    height = lat_h * model.vae_stride[1]
    width = lat_w * model.vae_stride[2]
    video = resize_frames(frames, height, width)
    latent = vae_encode_video_tiled(model.vae, video)
    seq_len = compute_seq_len(model, latent.shape)
    metadata = {
        "size": f"{width}*{height}",
        "latent_shape": tuple(latent.shape),
        "input_frame_count": input_frame_count,
    }
    return latent, seq_len, metadata


def resize_latent_spatial(latent, target_h, target_w, dtype):
    if latent.shape[-2:] == (target_h, target_w):
        return latent
    z = latent.permute(1, 0, 2, 3).float()
    z = F.interpolate(
        z,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False)
    return z.permute(1, 0, 2, 3).to(dtype=dtype)


def resize_latent_to_size(model, latent, reference_size, target_size):
    reference_width, reference_height = parse_size(reference_size)
    target_width, target_height = parse_size(target_size)
    target_area = target_width * target_height
    target_lat_h, target_lat_w = best_latent_size(
        reference_width, reference_height, target_area, model.vae_stride,
        model.patch_size)
    resized = resize_latent_spatial(latent, target_lat_h, target_lat_w,
                                    model.vae.dtype)
    decoded_height = target_lat_h * model.vae_stride[1]
    decoded_width = target_lat_w * model.vae_stride[2]
    return resized, {
        "size": f"{decoded_width}*{decoded_height}",
        "latent_shape": tuple(resized.shape),
    }


def encode_video_inputs_latent_resize(model, video_path, frame_num, size=None):
    frames, input_frame_count = read_video(video_path, frame_num)
    return encode_frames_latent_resize(model, frames, input_frame_count, size=size)


def encode_frames_latent_resize(model, frames, input_frame_count, size=None):
    source_area = frames[0].width * frames[0].height
    if size is None:
        target_area = source_area
    else:
        requested_width, requested_height = parse_size(size)
        target_area = requested_width * requested_height

    source_lat_h, source_lat_w = best_latent_size(
        frames[0].width, frames[0].height, source_area, model.vae_stride,
        model.patch_size)
    target_lat_h, target_lat_w = best_latent_size(
        frames[0].width, frames[0].height, target_area, model.vae_stride,
        model.patch_size)

    source_height = source_lat_h * model.vae_stride[1]
    source_width = source_lat_w * model.vae_stride[2]
    target_height = target_lat_h * model.vae_stride[1]
    target_width = target_lat_w * model.vae_stride[2]

    video = resize_frames(frames, source_height, source_width)
    source_latent = vae_encode_video_tiled(model.vae, video)
    latent = resize_latent_spatial(source_latent, target_lat_h, target_lat_w,
                                   model.vae.dtype)
    seq_len = compute_seq_len(model, latent.shape)
    metadata = {
        "size": f"{target_width}*{target_height}",
        "encoded_size": f"{source_width}*{source_height}",
        "video_resize_mode": "latent",
        "latent_shape": tuple(latent.shape),
        "encoded_latent_shape": tuple(source_latent.shape),
        "input_frame_count": input_frame_count,
    }
    return latent, seq_len, metadata


def create_prompt_noise_latent(model, size, frame_num):
    width, height = parse_size(size)
    latent_h, latent_w = best_latent_size(
        width, height, width * height, model.vae_stride, model.patch_size)
    decoded_height = latent_h * model.vae_stride[1]
    decoded_width = latent_w * model.vae_stride[2]
    latent_frames = (frame_num - 1) // model.vae_stride[0] + 1
    latent = torch.randn(
        model.vae.model.z_dim,
        latent_frames,
        latent_h,
        latent_w,
        dtype=torch.float32,
        device=model.device)
    seq_len = compute_seq_len(model, latent.shape)
    metadata = {
        "size": f"{decoded_width}*{decoded_height}",
        "video_resize_mode": "prompt_direct",
        "latent_shape": tuple(latent.shape),
        "input_frame_count": frame_num,
    }
    return latent, seq_len, metadata


def add_noise_to_clean_latent(clean_latent, sigma, noise):
    sigma = sigma.to(clean_latent.device).float()
    noisy = (1.0 - sigma) * clean_latent.float() + sigma * noise.float()
    return noisy.to(dtype=clean_latent.dtype).detach()


def start_index_for_step_count(round_noise_steps, sample_steps):
    if round_noise_steps < 1:
        raise ValueError("--round_noise_steps must be at least 1.")
    if round_noise_steps > sample_steps:
        raise ValueError("--round_noise_steps cannot exceed sample_steps.")
    return sample_steps - round_noise_steps


def denoise(model,
            clean_latent,
            round_noise_steps,
            context,
            context_null,
            seq_len,
            timesteps,
            sigmas,
            sample_steps,
            guide_scale,
            offload_model,
            y=None):
    
     
    random_noise = torch.randn_like(clean_latent)
    
    start_index = start_index_for_step_count(
        round_noise_steps,
        sample_steps,
    )
    
    noisy_latent =  add_noise_to_clean_latent(
                    clean_latent.detach(),
                    sigmas[start_index],
                    random_noise
                )
    output_latent ,_ = denoise_trajectory(
        model=model,
        start_latent = noisy_latent.detach(),
        context=context,
        context_null=context_null,
        seq_len=seq_len,
        timesteps=timesteps,
        sigmas=sigmas,
        sample_steps=sample_steps,
        start_index=start_index,
        guide_scale=guide_scale,
        offload_model=offload_model,
        y=y
    )

    return output_latent.detach(), start_index

def denoise_trajectory(model,
                       start_latent,
                       context,
                       context_null,
                       seq_len,
                       timesteps,
                       sigmas,
                       sample_steps,
                       start_index,
                       guide_scale,
                       offload_model,
                       step_callback=None,
                       y=None):
    boundary = model.boundary * model.num_train_timesteps
    latent = start_latent.detach()

    for i in tqdm(
            range(start_index, sample_steps),
            desc="Denoising",
            disable=not is_main_process()):
        if step_callback is not None:
            step_callback(i)
        with torch.no_grad(), torch.amp.autocast("cuda",
                                                 dtype=model.param_dtype):
            flow, _, _ = predict_cond_uncond(
                model,
                latent,
                timesteps[i],
                context,
                context_null,
                seq_len,
                guide_scale,
                boundary,
                offload_model,
                y=y)
            latent = flow_euler_step(latent, flow, sigmas[i],
                                     sigmas[i + 1]).detach()

    return latent, flow


def save_latent_video_streaming(vae, latent, save_path, fps, temporal_pad=0):
    return save_latent_video_tiled(vae, latent, save_path, fps,
                                   temporal_pad=temporal_pad)


def decode_latent_to_video_tiled(vae,
                                 latent,
                                 tile_h=192,
                                 tile_w=192,
                                 overlap=32):
    if overlap >= min(tile_h, tile_w):
        raise ValueError("VAE decode tile overlap must be smaller than tile size.")

    vae_model = vae.model
    device = vae.device
    z = latent.to(device).unsqueeze(0)

    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=vae.dtype):
            if isinstance(vae.scale[0], torch.Tensor):
                z = z / vae.scale[1].view(1, vae_model.z_dim, 1, 1, 1)
                z = z + vae.scale[0].view(1, vae_model.z_dim, 1, 1, 1)
            else:
                z = z / vae.scale[1] + vae.scale[0]
            x = vae_model.conv2(z)

            _, _, frames, height, width = x.shape
            stride_h = tile_h - overlap
            stride_w = tile_w - overlap
            ys = _tile_starts(height, tile_h, stride_h)
            xs = _tile_starts(width, tile_w, stride_w)

            vae_model.clear_cache()
            tile_caches = {
                (y0, x0): [None] * vae_model._conv_num
                for y0 in ys
                for x0 in xs
            }
            decoded_frames = []
            total_tiles = frames * len(ys) * len(xs)
            pbar = tqdm(
                total=total_tiles,
                desc="VAE Decode",
                unit="tile",
                disable=not is_main_process())

            try:
                for i in range(frames):
                    pbar.set_postfix(frame=f"{i + 1}/{frames}")
                    out_canvas = None
                    weight_canvas = None
                    scale_h = None
                    scale_w = None

                    for y0 in ys:
                        for x0 in xs:
                            y1 = min(y0 + tile_h, height)
                            x1 = min(x0 + tile_w, width)
                            tile = x[:, :, i:i + 1, y0:y1, x0:x1]

                            tile_cache = tile_caches[(y0, x0)]
                            vae_model._conv_idx = [0]
                            tile_out = vae_model.decoder(
                                tile,
                                feat_cache=tile_cache,
                                feat_idx=vae_model._conv_idx).float()
                            for cache_index, cached in enumerate(tile_cache):
                                if isinstance(cached, torch.Tensor):
                                    tile_cache[cache_index] = cached.cpu()

                            _, out_channels, out_t, out_h, out_w = tile_out.shape
                            if out_canvas is None:
                                scale_h = out_h / (y1 - y0)
                                scale_w = out_w / (x1 - x0)
                                full_h = round(height * scale_h)
                                full_w = round(width * scale_w)
                                out_canvas = torch.zeros(
                                    1,
                                    out_channels,
                                    out_t,
                                    full_h,
                                    full_w,
                                    device=device,
                                    dtype=torch.float32)
                                weight_canvas = torch.zeros_like(out_canvas)

                            out_y0 = round(y0 * scale_h)
                            out_x0 = round(x0 * scale_w)
                            out_y1 = out_y0 + out_h
                            out_x1 = out_x0 + out_w

                            weight_y = torch.hann_window(
                                out_h, periodic=False,
                                device=device).clamp_min(1e-3)
                            weight_x = torch.hann_window(
                                out_w, periodic=False,
                                device=device).clamp_min(1e-3)
                            weight = (weight_y[:, None] *
                                      weight_x[None, :]).view(
                                          1, 1, 1, out_h, out_w)

                            out_canvas[:, :, :, out_y0:out_y1,
                                       out_x0:out_x1] += tile_out * weight
                            weight_canvas[:, :, :, out_y0:out_y1,
                                          out_x0:out_x1] += weight

                            del tile, tile_out, weight
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            pbar.update(1)

                    out = (out_canvas / weight_canvas.clamp_min(1e-6)).clamp_(
                        -1, 1).squeeze(0)
                    decoded_frames.extend(frame.cpu() for frame in out.unbind(1))

                    del out_canvas, weight_canvas, out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                pbar.close()

    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if tile_h <= 48 or tile_w <= 48:
            vae_model.clear_cache()
            raise
        logging.warning(
            "Tiled VAE decode OOM at %dx%d latent tiles. Retrying with %dx%d tiles.",
            tile_h, tile_w, max(48, tile_h // 2), max(48, tile_w // 2))
        vae_model.clear_cache()
        return decode_latent_to_video_tiled(
            vae,
            latent,
            tile_h=max(48, tile_h // 2),
            tile_w=max(48, tile_w // 2),
            overlap=min(12, max(0, max(48, tile_h // 2) - 1),
                        max(0, max(48, tile_w // 2) - 1)))
    finally:
        vae_model.clear_cache()

    return torch.stack(decoded_frames, dim=1)


def save_latent_video_tiled(vae,
                            latent,
                            save_path,
                            fps,
                            tile_h=192,
                            tile_w=192,
                            overlap=32,
                            temporal_pad=0):
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "Saving videos requires imageio. Install Wan2.2 requirements in "
            "the active environment, e.g. `pip install -r Wan2.2/requirements.txt`."
        ) from exc

    if temporal_pad < 0:
        raise ValueError("--decode_temporal_pad must be non-negative.")
    if overlap >= min(tile_h, tile_w):
        raise ValueError("VAE decode tile overlap must be smaller than tile size.")

    vae_model = vae.model
    device = vae.device
    z = latent.to(device).unsqueeze(0)

    if temporal_pad > 0:
        pad = z[:, :, :1].repeat(1, 1, temporal_pad, 1, 1)
        z = torch.cat([pad, z], dim=2)

    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=vae.dtype):
            if isinstance(vae.scale[0], torch.Tensor):
                z = z / vae.scale[1].view(1, vae_model.z_dim, 1, 1, 1)
                z = z + vae.scale[0].view(1, vae_model.z_dim, 1, 1, 1)
            else:
                z = z / vae.scale[1] + vae.scale[0]
            x = vae_model.conv2(z)

            _, _, frames, height, width = x.shape
            stride_h = tile_h - overlap
            stride_w = tile_w - overlap
            ys = list(range(0, height, stride_h))
            xs = list(range(0, width, stride_w))
            ys[-1] = max(0, height - tile_h)
            xs[-1] = max(0, width - tile_w)
            ys = sorted(set(ys))
            xs = sorted(set(xs))

            vae_model.clear_cache()
            tile_caches = {
                (y0, x0): [None] * vae_model._conv_num
                for y0 in ys
                for x0 in xs
            }
            written_frames = 0
            total_tiles = frames * len(ys) * len(xs)
            pbar = tqdm(
                total=total_tiles,
                desc="VAE Decode",
                unit="tile",
                disable=not is_main_process())

            try:
                for i in range(frames):
                    pbar.set_postfix(frame=f"{i + 1}/{frames}")
                    out_canvas = None
                    weight_canvas = None
                    scale_h = None
                    scale_w = None

                    for y0 in ys:
                        for x0 in xs:
                            y1 = min(y0 + tile_h, height)
                            x1 = min(x0 + tile_w, width)
                            tile = x[:, :, i:i + 1, y0:y1, x0:x1]

                            tile_cache = tile_caches[(y0, x0)]
                            vae_model._conv_idx = [0]
                            tile_out = vae_model.decoder(
                                tile,
                                feat_cache=tile_cache,
                                feat_idx=vae_model._conv_idx).float()
                            for cache_index, cached in enumerate(tile_cache):
                                if isinstance(cached, torch.Tensor):
                                    tile_cache[cache_index] = cached.cpu()

                            _, out_channels, out_t, out_h, out_w = tile_out.shape
                            if out_canvas is None:
                                scale_h = out_h / (y1 - y0)
                                scale_w = out_w / (x1 - x0)
                                full_h = round(height * scale_h)
                                full_w = round(width * scale_w)
                                out_canvas = torch.zeros(
                                    1,
                                    out_channels,
                                    out_t,
                                    full_h,
                                    full_w,
                                    device=device,
                                    dtype=torch.float32)
                                weight_canvas = torch.zeros_like(out_canvas)

                            out_y0 = round(y0 * scale_h)
                            out_x0 = round(x0 * scale_w)
                            out_y1 = out_y0 + out_h
                            out_x1 = out_x0 + out_w

                            weight_y = torch.hann_window(
                                out_h, periodic=False,
                                device=device).clamp_min(1e-3)
                            weight_x = torch.hann_window(
                                out_w, periodic=False,
                                device=device).clamp_min(1e-3)
                            weight = (weight_y[:, None] *
                                      weight_x[None, :]).view(
                                          1, 1, 1, out_h, out_w)

                            out_canvas[:, :, :, out_y0:out_y1,
                                       out_x0:out_x1] += tile_out * weight
                            weight_canvas[:, :, :, out_y0:out_y1,
                                          out_x0:out_x1] += weight

                            del tile, tile_out, weight
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            pbar.update(1)

                    out = (out_canvas / weight_canvas.clamp_min(1e-6)).clamp_(
                        -1, 1).squeeze(0)
                    for frame in out.unbind(1):
                        if written_frames < temporal_pad:
                            written_frames += 1
                            continue
                        frame = ((frame + 1.0) * 127.5).clamp_(0, 255)
                        frame = frame.to(torch.uint8).permute(
                            1, 2, 0).cpu().numpy()
                        writer.append_data(frame)
                        written_frames += 1

                    del out_canvas, weight_canvas, out
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            finally:
                pbar.close()

    except torch.OutOfMemoryError:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if tile_h <= 48 or tile_w <= 48:
            writer.close()
            vae_model.clear_cache()
            raise
        logging.warning(
            "Tiled VAE decode OOM at %dx%d latent tiles. Retrying with %dx%d tiles.",
            tile_h, tile_w, max(48, tile_h // 2), max(48, tile_w // 2))
        writer.close()
        vae_model.clear_cache()
        return save_latent_video_tiled(
            vae,
            latent,
            save_path,
            fps,
            tile_h=max(48, tile_h // 2),
            tile_w=max(48, tile_w // 2),
            overlap=min(12, max(0, max(48, tile_h // 2) - 1),
                        max(0, max(48, tile_w // 2) - 1)),
            temporal_pad=temporal_pad)
    finally:
        writer.close()
        vae_model.clear_cache()


def save_latent_video_streaming_full_frame(vae,
                                           latent,
                                           save_path,
                                           fps,
                                           temporal_pad=0):
    try:
        import imageio
    except ImportError as exc:
        raise ImportError(
            "Saving videos requires imageio. Install Wan2.2 requirements in "
            "the active environment, e.g. `pip install -r Wan2.2/requirements.txt`."
        ) from exc

    vae_model = vae.model
    z = latent.to(vae.device)
    if temporal_pad < 0:
        raise ValueError("--decode_temporal_pad must be non-negative.")
    writer = imageio.get_writer(save_path, fps=fps, codec="libx264", quality=8)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=vae.dtype):
            vae_model.clear_cache()
            z = z.unsqueeze(0)
            if temporal_pad > 0:
                pad = z[:, :, :1].repeat(1, 1, temporal_pad, 1, 1)
                z = torch.cat([pad, z], dim=2)
            if isinstance(vae.scale[0], torch.Tensor):
                z = z / vae.scale[1].view(1, vae_model.z_dim, 1, 1, 1)
                z = z + vae.scale[0].view(1, vae_model.z_dim, 1, 1, 1)
            else:
                z = z / vae.scale[1] + vae.scale[0]
            x = vae_model.conv2(z)
            written_frames = 0
            for i in range(x.shape[2]):
                vae_model._conv_idx = [0]
                out = vae_model.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=vae_model._feat_map,
                    feat_idx=vae_model._conv_idx)
                out = out.float().clamp_(-1, 1).squeeze(0)
                for frame in out.unbind(1):
                    if written_frames < temporal_pad:
                        written_frames += 1
                        continue
                    frame = ((frame + 1.0) * 127.5).clamp_(0, 255)
                    frame = frame.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
                    writer.append_data(frame)
                    written_frames += 1
                del out
                torch.cuda.empty_cache()
    finally:
        writer.close()
        vae_model.clear_cache()


def load_latent_payload(path):
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, torch.Tensor):
        return payload, {}
    if "final_latent" in payload:
        return payload["final_latent"], payload.get("metadata", {})
    if "clean_latent" in payload:
        return payload["clean_latent"], payload.get("metadata", {})
    raise KeyError(
        f"{path} must contain a tensor, 'final_latent', or 'clean_latent'.")


def decode_latent_only(args, cfg):
    if args.save_video is None:
        raise ValueError("--decode_latent requires --save_video.")

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", args.device_id))
    if rank != 0:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    from wan.modules.vae2_1 import Wan2_1_VAE

    latent, metadata = load_latent_payload(args.decode_latent)
    fps = metadata.get("fps", args.fps)
    vae_dtype = parse_torch_dtype(args.vae_dtype)
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(args.ckpt_dir, cfg.vae_checkpoint),
        dtype=vae_dtype,
        device=torch.device(f"cuda:{local_rank}"))
    set_vae_dtype(vae, vae_dtype)
    save_latent_video_streaming(vae, latent, args.save_video, fps,
                                args.decode_temporal_pad)
    logging.info("Decoded %s to %s", args.decode_latent, args.save_video)


def run(args, model, cfg):
    if args.video is None and args.prompt is None:
        raise ValueError(
            "--prompt is required when --video is not provided.")
    if (args.prompt is None):
        raise ValueError("--prompt is required unless --decode_latent is set.")

    frame_num = args.frame_num or cfg.frame_num
    sample_steps = args.sample_steps or cfg.sample_steps
    sample_shift = args.sample_shift or cfg.sample_shift
    guide_scale = args.sample_guide_scale or cfg.sample_guide_scale
    guide_scale = (guide_scale, guide_scale) if isinstance(
        guide_scale, float) else guide_scale
    step_offload_model = args.offload_model and not args.dit_fsdp
    text_offload_model = args.offload_model and not args.t5_fsdp

    if frame_num % 4 != 1:
        raise ValueError("--frame_num must be 4n+1 for Wan T2V.")
    if args.noise_rounds < 1:
        raise ValueError("--noise_rounds must be at least 1.")

    if args.offload_model and args.video is not None:
        offload_dit_models(model)


    encode_size = args.size

    context, context_null = prepare_text_context(model, args.prompt,
                                                     args.negative_prompt,
                                                     text_offload_model)
    timesteps, sigmas = make_scheduler(model, sample_steps,
                                                    sample_shift)

    if args.video is None:
        prompt_base_size = "1280*720"
        base_latent, base_seq_len, _ = create_prompt_noise_latent(
            model, prompt_base_size, frame_num)
        if is_main_process():
            logging.info("Generating prompt-only base latent at %s",
                         prompt_base_size)
        configure_block_tiled_attention_for_base = False
        set_block_tiled_self_attention(
            model,
            configure_block_tiled_attention_for_base,
            args.block_tiled_self_attn_tile_height,
            args.block_tiled_self_attn_tile_width,
            args.block_tiled_self_attn_stride_height,
            args.block_tiled_self_attn_stride_width,
            args.block_tiled_self_attn_halo,
            args.block_tiled_self_attn_routed_topk,
            args.block_tiled_self_attn_routed_grid,
            args.block_tiled_self_attn_global_attention_mode,
            args.block_tiled_self_attn_global_rope_threshold,
            args.block_tiled_self_attn_blocks)
        base_latent, _ = denoise_trajectory(
            model=model,
            start_latent=base_latent,
            context=context,
            context_null=context_null,
            seq_len=base_seq_len,
            timesteps=timesteps,
            sigmas=sigmas,
            sample_steps=sample_steps,
            start_index=0,
            guide_scale=guide_scale,
            offload_model=step_offload_model,
            y=None)

        if args.video_resize_mode == "latent":
            clean_latent, resize_metadata = resize_latent_to_size(
                model, base_latent, prompt_base_size, encode_size)
            seq_len = compute_seq_len(model, clean_latent.shape)
            metadata = {
                **resize_metadata,
                "prompt_base_size": prompt_base_size,
                "prompt_base_latent_shape": tuple(base_latent.shape),
                "video_resize_mode": "prompt_latent",
                "input_frame_count": frame_num,
            }
        elif args.video_resize_mode == "pixel":
            if args.offload_model:
                offload_dit_models(model)
                model.vae.model.to(device=model.vae.device, dtype=model.vae.dtype)
            base_video = decode_latent_to_video_tiled(model.vae, base_latent)
            target_width, target_height = parse_size(encode_size)
            target_lat_h, target_lat_w = best_latent_size(
                target_width, target_height, target_width * target_height,
                model.vae_stride, model.patch_size)
            target_height = target_lat_h * model.vae_stride[1]
            target_width = target_lat_w * model.vae_stride[2]
            base_video = F.interpolate(
                base_video.permute(1, 0, 2, 3).float(),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False).permute(1, 0, 2, 3).contiguous()
            clean_latent = vae_encode_video_tiled(model.vae, base_video)
            seq_len = compute_seq_len(model, clean_latent.shape)
            metadata = {
                "size": f"{target_width}*{target_height}",
                "prompt_base_size": prompt_base_size,
                "prompt_base_latent_shape": tuple(base_latent.shape),
                "video_resize_mode": "prompt_pixel",
                "latent_shape": tuple(clean_latent.shape),
                "input_frame_count": frame_num,
            }
            del base_video
            if args.offload_model:
                offload_vae_model(model)
                onload_dit_models(model)
        else:
            raise ValueError("--video_resize_mode must be 'pixel' or 'latent'.")
        del base_latent
        metadata["input_mode"] = "prompt"
    elif args.video_resize_mode == "pixel":
        clean_latent, seq_len, metadata = encode_video_inputs(
            model, args.video, frame_num, size=encode_size)
        metadata["video_resize_mode"] = "pixel"
    elif args.video_resize_mode == "latent":
        clean_latent, seq_len, metadata = encode_video_inputs_latent_resize(
            model, args.video, frame_num, size=encode_size)
    else:
        raise ValueError("--video_resize_mode must be 'pixel' or 'latent'.")
    if is_main_process():
        if args.video is None:
            logging.info("Generated prompt base at %s, then prepared %s latent at %s",
                         metadata["prompt_base_size"],
                         metadata["video_resize_mode"],
                         metadata["size"])
        else:
            logging.info("Encoded video at %s", metadata["size"])
    
        if metadata.get("video_resize_mode") == "latent":
            logging.info("Encoded source video at %s, then resized latent to %s",
                         metadata["encoded_size"], metadata["size"])

    if args.offload_model:
        offload_vae_model(model)

    if args.offload_model and (args.t5_fsdp or args.dit_fsdp):
        onload_dit_models(model)

    set_self_attention_scale(model, args.self_attn_scale)
           

    def configure_block_tiled_attention(enabled):
        set_block_tiled_self_attention(
            model,
            enabled,
            args.block_tiled_self_attn_tile_height,
            args.block_tiled_self_attn_tile_width,
            args.block_tiled_self_attn_stride_height,
            args.block_tiled_self_attn_stride_width,
            args.block_tiled_self_attn_halo,
            args.block_tiled_self_attn_routed_topk,
            args.block_tiled_self_attn_routed_grid,
            args.block_tiled_self_attn_global_attention_mode,
            args.block_tiled_self_attn_global_rope_threshold,
            args.block_tiled_self_attn_blocks)

    configure_block_tiled_attention(args.block_tiled_self_attn)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low = getattr(model.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high = (
        noop_no_sync if model.high_noise_model is model.low_noise_model else
        getattr(model.high_noise_model, "no_sync", noop_no_sync))
    latent_metadata = {
        "video": args.video,
        "input_mode": metadata.get("input_mode", "video"),
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "size": metadata["size"],
        "prompt_base_size": metadata.get("prompt_base_size"),
        "prompt_base_latent_shape": metadata.get("prompt_base_latent_shape"),
        "video_resize_mode": metadata.get("video_resize_mode", "pixel"),
        "encoded_size": metadata.get("encoded_size"),
        "encoded_latent_shape": metadata.get("encoded_latent_shape"),
        "frame_num": frame_num,
        "fps": args.fps,
        "sample_steps": sample_steps,
        "sample_shift": sample_shift,
        "sample_guide_scale": guide_scale,
        "round_noise_steps": args.round_noise_steps,
        "noise_rounds": args.noise_rounds,
        "model_version": args.model_version,
        "ckpt_dir": args.ckpt_dir,
        "block_tiled_self_attn": args.block_tiled_self_attn,
        "block_tiled_self_attn_routed_topk": (
            args.block_tiled_self_attn_routed_topk),
        "block_tiled_self_attn_routed_grid": (
            args.block_tiled_self_attn_routed_grid),
        "block_tiled_self_attn_global_attention_mode": (
            args.block_tiled_self_attn_global_attention_mode),
        "block_tiled_self_attn_global_rope_threshold": (
            args.block_tiled_self_attn_global_rope_threshold),
    }

    restart_metadata = []
    final_latent = clean_latent.detach()
    with no_sync_low(), no_sync_high():
        for restart_idx in range(args.noise_rounds):
            if is_main_process():
                logging.info("Noise restart %d/%d: adding noise and denoising %d steps",
                             restart_idx + 1, args.noise_rounds,
                             args.round_noise_steps)
            final_latent, start_index = denoise(
                model=model,
                clean_latent=final_latent,
                round_noise_steps=args.round_noise_steps,
                context=context,
                context_null=context_null,
                seq_len=seq_len,
                timesteps=timesteps,
                sigmas=sigmas,
                sample_steps=sample_steps,
                guide_scale=guide_scale,
                offload_model=step_offload_model,
                y=None)
            restart_metadata.append({
                "round": restart_idx + 1,
                "start_index": start_index,
                "start_sigma": float(sigmas[start_index].detach().cpu()),
            })
    if args.offload_model:  
        offload_dit_models(model)

    if args.save_latent is not None and is_main_process():
        payload = {
            "final_latent": final_latent.detach().cpu(),
            "metadata": {
                **latent_metadata,
                "latent_shape": tuple(final_latent.shape),
                "start_index": start_index,
                "start_sigma": float(sigmas[start_index].detach().cpu()),
                "noise_restarts": restart_metadata,
            },
        }
        Path(args.save_latent).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.save_latent)
        logging.info("Saved latent to %s", args.save_latent)

    if args.save_video is not None and is_main_process():
        Path(args.save_video).parent.mkdir(parents=True, exist_ok=True)
        model.vae.model.to(device=model.vae.device, dtype=model.vae.dtype)
        save_latent_video_streaming(model.vae, final_latent, args.save_video,
                                    args.fps, args.decode_temporal_pad)
        logging.info("Decoded final latent to %s", args.save_video)

    del clean_latent, context, context_null
    del timesteps, sigmas
    gc.collect()
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan2.2 T2V video upscaling with noise-based denoising experiments."
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional input video path. If omitted, a 1280*720 prompt-only latent is generated first and then latent-upsampled to --size.")
    parser.add_argument("--prompt", default=None, help="T2V prompt.")
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Wan2.2-T2V-A14B checkpoint directory.")
    
    parser.add_argument("--save_video", default=None, help="Optional decoded output mp4 path.")
    parser.add_argument("--save_latent", default=None, help="Optional final latent .pt path.")
    parser.add_argument(
        "--decode_latent",
        default=None,
        help="Decode this saved latent .pt and exit without loading DiT/T5.")

    parser.add_argument(
        "--vae_dtype",
        default="fp16",
        choices=("fp32", "fp16", "bf16"),
        help="VAE encode/decode dtype.")
    parser.add_argument(
        "--decode_temporal_pad",
        type=int,
        default=0,
        help="Duplicate this many first latent slices before VAE decode and skip their decoded frames.")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument(
        "--size",
        default="1920*1080",
        help="Target area as width*height. Aspect ratio follows input video.")
    parser.add_argument(
        "--video_resize_mode",
        default="pixel",
        choices=("pixel", "latent"),
        help="pixel: resize frames to --size before VAE encoding. latent: encode at input/native size first, then bilinearly resize the VAE latent to --size.")

    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument(
        "--round_noise_steps",
        type=int,
        default=1,
        help="Exact denoising steps per noise round.")
    parser.add_argument(
        "--noise_rounds",
        type=int,
        default=1,
        help="Number of repeated noise restart cycles. Each cycle adds fresh noise to the current latent and denoises --round_noise_steps steps.")
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--self_attn_scale",
        type=float,
        default=1.0,
        help="Multiplier s for DiT self-attention logits. Uses s / sqrt(head_dim).")
    parser.add_argument(
        "--anchor_attn_local_window",
        type=int,
        default=8,
        help="Spatial token window for local high-resolution self-attention.")
    parser.add_argument(
        "--anchor_attn_local_halo",
        type=int,
        default=0,
        help="Extra spatial token border added to local K/V windows. 0 keeps non-overlapping windows.")
    parser.add_argument(
        "--anchor_attn_blocks",
        default="",
        help="Comma/range block selector for anchor attention, e.g. '24-39'. Empty means all blocks.")
    parser.add_argument(
        "--block_tiled_self_attn",
        type=str2bool,
        default=False,
        help="Tile only DiT self-attention inside each block, stitch the self-attention output, then run global cross-attention/FFN.")
    parser.add_argument(
        "--block_tiled_self_attn_tile_height",
        type=int,
        default=45,
        help="Inner self-attention tile height in transformer patch-token units.")
    parser.add_argument(
        "--block_tiled_self_attn_tile_width",
        type=int,
        default=78,
        help="Inner self-attention tile width in transformer patch-token units.")
    parser.add_argument(
        "--block_tiled_self_attn_stride_height",
        type=int,
        default=24,
        help="Self-attention tile stride height in transformer patch-token units.")
    parser.add_argument(
        "--block_tiled_self_attn_stride_width",
        type=int,
        default=42,
        help="Self-attention tile stride width in transformer patch-token units.")
    parser.add_argument(
        "--block_tiled_self_attn_halo",
        type=int,
        default=6,
        help="Halo context in transformer patch-token units for blockwise tiled self-attention.")
    parser.add_argument(
        "--block_tiled_self_attn_routed_topk",
        type=int,
        default=0,
        help="Retrieve this many content-routed global grids per tile. 0 disables routed global K/V.")
    parser.add_argument(
        "--block_tiled_self_attn_routed_grid",
        type=int,
        default=3,
        help="Spatial grid size, in transformer patch tokens, for content-routed global retrieval.")
    parser.add_argument(
        "--block_tiled_self_attn_global_attention_mode",
        choices=("separate", "joint"),
        default="separate",
        help="Use separate local/global attention outputs or one joint softmax over local plus global K/V.")
    parser.add_argument(
        "--block_tiled_self_attn_global_rope_threshold",
        type=float,
        default=24.0,
        help="Uncompressed local distance threshold before geometry-derived compressed-relative global RoPE.")
    parser.add_argument(
        "--block_tiled_self_attn_blocks",
        default="",
        help="Comma/range block selector for blockwise tiled self-attention. Empty means all blocks.")
    parser.add_argument("--base_seed", type=int, default=-1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="Sequence-parallel world size for DiT. With torchrun, set this to nproc_per_node."
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Shard the T5 text encoder with FSDP in distributed runs.")
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Shard the Wan DiT models with FSDP in distributed runs.")
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    import wan
    from wan.configs import WAN_CONFIGS

    cfg = make_model_config(WAN_CONFIGS)
    args.model_version = "2.2"
    logging.info("Using Wan2.2 checkpoint layout")
    if args.decode_latent is not None:
        decode_latent_only(args, cfg)
        return

    rank, world_size, _ = setup_distributed(args)

    if args.offload_model is None:
        args.offload_model = True
        if is_main_process():
            logging.info("offload_model not specified; using %s",
                         args.offload_model)

    if args.ulysses_size > 1 and cfg.num_heads % args.ulysses_size != 0:
        raise ValueError(
            f"cfg.num_heads={cfg.num_heads} must be divisible by --ulysses_size."
        )

    seed = args.base_seed if args.base_seed >= 0 else (
        random.randint(0, sys.maxsize) if rank == 0 else 0)
    if dist.is_initialized():
        seed_holder = [seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed_holder, src=0)
        seed = seed_holder[0]
    random.seed(seed)
    torch.manual_seed(seed)
    args.run_seed = seed


    model = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=args.device_id,
        rank=args.rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype)
    model.model_version = "2.2"

    set_vae_dtype(model.vae, parse_torch_dtype(args.vae_dtype))
    run(args, model, cfg)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


# 3840*2160 2560*1440

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# python CineScale/Wan2.2/cinescale.py \
#   --decode_latent CineScale/latent_result.pt \
#   --ckpt_dir Wan2.2-T2V-A14B \
#   --save_video CineScale/result_video.mp4

#   --video CineScale/Eiffel_Tower.mp4 \

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# torchrun --standalone --nproc_per_node=4 CineScale/Wan2.2/cinescale.py \
#   --size "3840*2160" \
#   --prompt "A tiny astronaut walking across the surface of a giant orange, macro photography, shallow depth of field." \
#   --ckpt_dir Wan2.2-T2V-A14B \
#   --frame_num 17 \
#  --video_resize_mode pixel \
#   --round_noise_steps 20 \
#   --sample_shift 12 \
#   --block_tiled_self_attn true \
#   --block_tiled_self_attn_tile_height 24 \
#   --block_tiled_self_attn_tile_width 24 \
#   --block_tiled_self_attn_stride_height 20 \
#   --block_tiled_self_attn_stride_width 20 \
#   --block_tiled_self_attn_halo 6 \
#  --block_tiled_self_attn_routed_topk 32 \
#   --block_tiled_self_attn_routed_grid 3 \
# --block_tiled_self_attn_global_attention_mode joint \
# --block_tiled_self_attn_global_rope_threshold 40 \
#   --save_latent CineScale/latent_result.pt \
# --decode_temporal_pad 4 \
# --noise_rounds 2 \
#   --ulysses_size 4 \
#   --dit_fsdp \
#   --t5_cpu \
#   --offload_model true
