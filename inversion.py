import argparse
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


def offload_dit_models(model):
    for name in ("low_noise_model", "high_noise_model"):
        dit_model = getattr(model, name, None)
        if dit_model is None:
            continue
        if next(dit_model.parameters()).device.type == "cuda":
            dit_model.to("cpu")
    torch.cuda.empty_cache()


def set_vae_dtype(vae, dtype):
    vae.dtype = dtype
    vae.mean = vae.mean.to(dtype=dtype, device=vae.device)
    vae.std = vae.std.to(dtype=dtype, device=vae.device)
    vae.scale = [vae.mean, 1.0 / vae.std]
    vae.model.to(device=vae.device, dtype=dtype)


def onload_dit_models(model):
    for name in ("low_noise_model", "high_noise_model"):
        dit_model = getattr(model, name, None)
        if dit_model is None:
            continue
        if next(dit_model.parameters()).device.type == "cpu":
            dit_model.to(model.device)
    torch.cuda.empty_cache()


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
    return scheduler, timesteps, scheduler.sigmas.to(model.device)


def flow_euler_step(latent, flow, sigma_from, sigma_to):
    return latent + (sigma_to - sigma_from) * flow


def set_wan_ntk(model, ntk_factor):
    if ntk_factor is None or ntk_factor <= 1:
        return

    from wan.modules.model import rope_params

    def set_model_ntk(wan_model):
        target_model = getattr(wan_model, "module", wan_model)
        head_dim = target_model.dim // target_model.num_heads
        frame_dim = head_dim - 4 * (head_dim // 6)
        spatial_dim = 2 * (head_dim // 6)
        freqs = torch.cat([
            rope_params(1024, frame_dim, theta=10000),
            rope_params(1024, spatial_dim, theta=10000 * ntk_factor),
            rope_params(1024, spatial_dim, theta=10000 * ntk_factor),
        ],
                          dim=1)
        target_model.freqs = freqs.to(next(target_model.parameters()).device)

    set_model_ntk(model.low_noise_model)
    set_model_ntk(model.high_noise_model)


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
    video = resize_frames(frames, height, width).to(model.device)
    latent = model.vae.encode([video])[0]
    seq_len = compute_seq_len(model, latent.shape)
    metadata = {
        "size": f"{width}*{height}",
        "latent_shape": tuple(latent.shape),
        "input_frame_count": input_frame_count,
    }
    return latent, seq_len, metadata



def add_noise_to_clean_latent(clean_latent, sigma, noise):
    sigma = sigma.to(clean_latent.device).float()
    noisy = (1.0 - sigma) * clean_latent.float() + sigma * noise.float()
    return noisy.to(dtype=clean_latent.dtype).detach()


def infer_flow_noise(noisy_latent, clean_latent, sigma):
    sigma = sigma.to(noisy_latent.device).float().clamp_min(1e-6)
    noise = (noisy_latent.float() -
             (1.0 - sigma) * clean_latent.float()) / sigma
    return noise






def lowpass_latent(latent, factor):
    if latent.ndim != 4:
        raise ValueError(
            f"Expected latent shape [C, T, H, W], got {tuple(latent.shape)}.")
    if factor <= 1:
        return latent

    _, latent_frames, lat_h, lat_w = latent.shape
    low_h = max(1, lat_h // factor)
    low_w = max(1, lat_w // factor)
    latent_low = F.interpolate(
        latent[None].float(),
        size=(latent_frames, low_h, low_w),
        mode="trilinear",
        align_corners=False)
    latent_low = F.interpolate(
        latent_low,
        size=(latent_frames, lat_h, lat_w),
        mode="trilinear",
        align_corners=False)[0]
    return latent_low.to(dtype=latent.dtype)


def principal_delta_projection(latents_delta_by_depth):
    if len(latents_delta_by_depth) < 1:
        raise ValueError("Need at least one latent delta for PCA projection.")

    reference = latents_delta_by_depth[0]
    D = torch.stack(
        [delta.float().flatten() for delta in latents_delta_by_depth],
        dim=0)
    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    explained = S[0].square() / S.square().sum().clamp_min(1e-12)
    u1 = Vh[0].reshape_as(reference)
    coeffs = [
        torch.sum(delta.float() * u1.float())
        for delta in latents_delta_by_depth
    ]
    projected_delta = coeffs[-1] * u1
    return projected_delta.to(dtype=reference.dtype), explained, coeffs


def intermediate_latent_path(args, round_number):
    if args.save_latent_dir is not None:
        return Path(args.save_latent_dir) / f"latent_round_{round_number:04d}.pt"
    if args.save_latent is not None:
        path = Path(args.save_latent)
        return path.with_name(f"{path.stem}_round_{round_number:04d}{path.suffix or '.pt'}")
    return Path(f"latent_round_{round_number:04d}.pt")


def maybe_save_round_latent(args, round_number, latent, metadata):
    if args.save_latent_every <= 0:
        return
    if round_number % args.save_latent_every != 0:
        return
    if not is_main_process():
        return

    path = intermediate_latent_path(args, round_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "final_latent": latent.detach().cpu(),
        "metadata": {
            **metadata,
            "round": round_number,
            "checkpoint_type": "intermediate_round",
        },
    }, path)
    logging.info("Saved round %s latent to %s", round_number, path)


def start_index_for_step_count(round_noise_steps, sample_steps):
    if round_noise_steps < 1:
        raise ValueError("--round_noise_steps must be at least 1.")
    if round_noise_steps > sample_steps:
        raise ValueError("--round_noise_steps cannot exceed sample_steps.")
    return sample_steps - round_noise_steps


def invert_with_trajectory(model,
                           clean_latent,
                           context,
                           context_null,
                           seq_len,
                           timesteps,
                           sigmas,
                           sample_steps,
                           invert_steps,
                           guide_scale,
                           offload_model,
                           fixed_point_iters=5,
                           y=None):
    if invert_steps < 1:
        raise ValueError("--invert_steps must be at least 1.")
    if invert_steps > sample_steps:
        raise ValueError("--invert_steps cannot exceed sample_steps.")
    if fixed_point_iters < 1:
        raise ValueError("--inversion_fixed_point_iters must be at least 1.")

    start_index = sample_steps - invert_steps
    boundary = model.boundary * model.num_train_timesteps
    # trajectory = {sample_steps: clean_latent.detach().cpu()}
    latent_next = clean_latent.detach()

    inverted_latents = []
    for i in tqdm(
            range(sample_steps - 1, start_index - 1, -1),
            desc="Inversion",
            disable=not is_main_process()):
        latent_guess = latent_next.detach()
        for _ in range(fixed_point_iters):
            with torch.no_grad(), torch.amp.autocast("cuda",
                                                     dtype=model.param_dtype):
                flow, _, _ = predict_cond_uncond(
                    model,
                    latent_guess,
                    timesteps[i],
                    context,
                    context_null,
                    seq_len,
                    guide_scale,
                    boundary,
                    offload_model,
                    y=y)
                delta = sigmas[i + 1] - sigmas[i]
                latent_guess = (latent_next - delta * flow).detach()
        latent_next = latent_guess.detach()
        inverted_latents.append(latent_next)

    return inverted_latents, start_index


def denoise(model,
            clean_latent,
            num_rounds,
            round_noise_steps,
            invert_steps,
            alpha,
            inversion_fixed_point_iters,
            context,
            context_null,
            seq_len,
            timesteps,
            sigmas,
            sample_steps,
            guide_scale,
            offload_model,
            round_callback=None,
            y=None):
    # if num_rounds < 1:
    #     raise ValueError("--noise_rounds must be at least 1.")
    
    beta = 0.1
    z0 = clean_latent.detach()
    # output_latent = clean_latent.detach()
    random_noise = torch.randn_like(clean_latent)

    latents_delta_by_depth = []
    
    for depth in range(round_noise_steps):
        start_index = start_index_for_step_count(depth+1, sample_steps)
        # for round_index in range(num_rounds):

        # previous_output = output_latent.detach()

        noisy_latent =  add_noise_to_clean_latent(
                        z0,
                        sigmas[start_index],
                        random_noise
                    )

        output_latent,_ = denoise_trajectory(
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
        latents_delta_by_depth.append(output_latent.detach() - z0)
    
        # if alpha != 1.0:
        #     output_latent = previous_output + alpha * (output_latent - previous_output)
        
        if round_callback is not None:
            round_callback(depth + 1, output_latent, z0,
                        start_index)
            
    print("\n=== Delta Cosine Similarities ===")

    for i in range(len(latents_delta_by_depth)):
        for j in range(i + 1, len(latents_delta_by_depth)):

            d1 = latents_delta_by_depth[i].float().flatten()
            d2 = latents_delta_by_depth[j].float().flatten()

            cos = F.cosine_similarity(
                d1.unsqueeze(0),
                d2.unsqueeze(0),
                dim=1
            ).item()

            print(f"d{i+1} vs d{j+1}: {cos:.4f}")

    print("\n=== Delta Norms ===")
    for i, d in enumerate(latents_delta_by_depth):
        print(f"||d{i+1}||: {d.float().norm().item():.6f}")

    print("\n=== Incremental Residual Cosine Similarities ===")

    residuals = []
    for i, d in enumerate(latents_delta_by_depth):
        if i == 0:
            residuals.append(d)
        else:
            residuals.append(d - latents_delta_by_depth[i - 1])

    for i in range(len(residuals)):
        for j in range(i + 1, len(residuals)):
            r1 = residuals[i].float().flatten()
            r2 = residuals[j].float().flatten()

            cos = F.cosine_similarity(
                r1.unsqueeze(0),
                r2.unsqueeze(0),
                dim=1
            ).item()

            print(f"r{i+1} vs r{j+1}: {cos:.4f}")
    
    print("\n=== Incremental Residual Norms ===")
    for i, r in enumerate(residuals):
        print(f"||r{i+1}||: {r.float().norm().item():.6f}")

    pca_delta, explained, coeffs = principal_delta_projection(
        latents_delta_by_depth)
    print("\n=== Delta PCA ===")
    print(f"pc1 explained variance: {explained.item():.8f}")
    for i, coeff in enumerate(coeffs):
        print(f"d{i+1} pc1 coeff: {coeff.item():.6f}")

    orthogonal_component = latents_delta_by_depth[-1] - pca_delta

    d_corrected = latents_delta_by_depth[0] - beta * orthogonal_component

    output_latent = (z0 + alpha * d_corrected).detach()
    print("Saving Final Latent depth 0")
    round_callback(0, output_latent, z0,start_index)

    return output_latent, z0, start_index

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
                       y=None):
    boundary = model.boundary * model.num_train_timesteps
    latent = start_latent.detach()

    for i in tqdm(
            range(start_index, sample_steps),
            desc="Denoising",
            disable=not is_main_process()):
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
    if args.video is None:
        raise ValueError("--video is required unless --decode_latent is set.")
    if args.prompt is None:
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
    set_wan_ntk(model, args.ntk_factor)

    if args.offload_model:
        offload_dit_models(model)

    clean_latent, seq_len, metadata = encode_video_inputs(
        model, args.video, frame_num, size=args.size)
    if is_main_process():
        logging.info("Encoded video at %s", metadata["size"])

    if args.encode_only:
        if args.save_latent is None:
            raise ValueError("--encode_only true requires --save_latent.")
        if is_main_process():
            payload = {
                "clean_latent": clean_latent.detach().cpu(),
                "final_latent": clean_latent.detach().cpu(),
                "metadata": {
                    "mode": "encode_only",
                    "video": args.video,
                    "size": metadata["size"],
                    "latent_shape": tuple(clean_latent.shape),
                    "frame_num": frame_num,
                    "fps": args.fps,
                    "input_frame_count": metadata["input_frame_count"],
                    "ckpt_dir": args.ckpt_dir,
                },
            }
            Path(args.save_latent).parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.save_latent)
            logging.info("Saved encoded latent to %s", args.save_latent)
        del clean_latent
        gc.collect()
        torch.cuda.empty_cache()
        return

    if args.offload_model and (args.t5_fsdp or args.dit_fsdp):
        onload_dit_models(model)
    context, context_null = prepare_text_context(model, args.prompt,
                                                 args.negative_prompt,
                                                 text_offload_model)
    scheduler, timesteps, sigmas = make_scheduler(model, sample_steps,
                                                  sample_shift)

    @contextmanager
    def noop_no_sync():
        yield

    no_sync_low = getattr(model.low_noise_model, "no_sync", noop_no_sync)
    no_sync_high = getattr(model.high_noise_model, "no_sync", noop_no_sync)
    latent_metadata = {
        "video": args.video,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "size": metadata["size"],
        "frame_num": frame_num,
        "fps": args.fps,
        "sample_steps": sample_steps,
        "sample_shift": sample_shift,
        "sample_guide_scale": guide_scale,
        "noise_rounds": args.noise_rounds,
        "round_noise_steps": args.round_noise_steps,
        "invert_steps": args.invert_steps,
        "alpha": args.alpha,
        "inversion_fixed_point_iters": args.inversion_fixed_point_iters,
        "ckpt_dir": args.ckpt_dir,
    }

    def save_round_callback(round_number, latent, _start_latent, round_start_index):
        round_metadata = {
            **latent_metadata,
            "latent_shape": tuple(latent.shape),
            "start_index": round_start_index,
            "start_sigma": float(sigmas[round_start_index].detach().cpu()),
        }
        maybe_save_round_latent(args, round_number, latent, round_metadata)

    with no_sync_low(), no_sync_high():
        
        final_latent, start_latent, start_index = denoise(
            model=model,
            clean_latent=clean_latent,
            num_rounds=args.noise_rounds,
            round_noise_steps=args.round_noise_steps,
            invert_steps=args.invert_steps,
            alpha=args.alpha,
            inversion_fixed_point_iters=args.inversion_fixed_point_iters,
            context=context,
            context_null=context_null,
            seq_len=seq_len,
            timesteps=timesteps,
            sigmas=sigmas,
            sample_steps=sample_steps,
            guide_scale=guide_scale,
            offload_model=step_offload_model,
            round_callback=save_round_callback,
            y=None)
        

    if args.save_latent is not None and is_main_process():
        payload = {
            "final_latent": final_latent.detach().cpu(),
            "metadata": {
                **latent_metadata,
                "latent_shape": tuple(final_latent.shape),
                "start_index": start_index,
                "start_sigma": float(sigmas[start_index].detach().cpu()),
            },
        }
        Path(args.save_latent).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.save_latent)
        logging.info("Saved latent to %s", args.save_latent)

    del scheduler, clean_latent, start_latent, context, context_null
    del timesteps, sigmas
    gc.collect()
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Wan2.2 T2V video upscaling with noise-based denoising experiments."
    )
    parser.add_argument("--video", default=None, help="Input video path.")
    parser.add_argument("--prompt", default=None, help="T2V prompt.")
    parser.add_argument(
        "--ckpt_dir",
        required=True,
        help="Wan2.2 T2V checkpoint directory.")
    
    parser.add_argument("--save_video", default=None, help="Optional decoded output mp4 path.")
    parser.add_argument("--save_latent", default=None, help="Optional final latent .pt path.")
    parser.add_argument(
        "--save_latent_every",
        type=int,
        default=0,
        help="Save an intermediate latent every N rounds. Use 0 to disable.")
    parser.add_argument(
        "--save_latent_dir",
        default=None,
        help="Optional directory for intermediate round latent checkpoints.")
    parser.add_argument(
        "--decode_latent",
        default=None,
        help="Decode this saved latent .pt and exit without loading DiT/T5.")
    parser.add_argument(
        "--encode_only",
        type=str2bool,
        default=False,
        help="Encode --video to --save_latent and exit without denoising.")
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
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--sample_steps", type=int, default=None)
    parser.add_argument(
        "--noise_rounds",
        type=int,
        default=1,
        help="Number of add-noise/denoise rounds.")
    parser.add_argument(
        "--round_noise_steps",
        type=int,
        default=1,
        help="Exact denoising steps per noise round.")
    parser.add_argument(
        "--invert_steps",
        type=int,
        default=1,
        help="Exact inversion steps before the final denoise pass.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Scale for proposal delta in --noise_mode detail_delta.")
    parser.add_argument(
        "--inversion_fixed_point_iters",
        type=int,
        default=5,
        help="Fixed-point iterations per implicit inversion step.")
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--sample_guide_scale", type=float, default=None)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument(
        "--ntk_factor",
        type=float,
        default=1.0,
        help="Spatial RoPE theta multiplier. Use 1.0 to disable.")
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

    cfg = WAN_CONFIGS["t2v-A14B"]
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
    set_vae_dtype(model.vae, parse_torch_dtype(args.vae_dtype))
    run(args, model, cfg)
    if is_main_process():
        if args.save_video is not None:
            logging.info("Saved video to %s", args.save_video)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()


# torchrun --nproc_per_node=1 CineScale/Wan2.2/cinescale.py \
#   --video CineScale/tokyo-walk-360p.mp4 \
#   --size "3840*2160" \
#   --prompt "" \
#   --ckpt_dir Wan2.2-T2V-A14B \
#   --frame_num 1 \
#   --save_latent CineScale/result_latent.pt \
#   --noise_rounds 1 \
#   --round_noise_steps 2 \
#   --inversion_fixed_point_iters 3 \
#   --invert_steps 2 \
#   --alpha 1.0 \
#   --sample_guide_scale 1 \
#   --offload_model true \
#   --ulysses_size 1 \
#   --save_latent_every 1 \
#   --save_latent_dir CineScale

#   --dit_fsdp \
#   --t5_fsdp \

# python CineScale/Wan2.2/cinescale.py \
#   --decode_latent CineScale/latent_round_0000.pt \
#   --ckpt_dir Wan2.2-T2V-A14B \
#   --save_video CineScale/result.mp4 


