# CineScale4K 🎬

CineScale4K is a Wan2.2-T2V inference sandbox for high-resolution video generation, video-guided high-resolution regeneration, and tiled high-resolution self-attention experiments.

The current code is centered on `Wan2.2/cinescale.py` and assumes Wan2.2 checkpoints.

## Novelty 🚀

This repo explores high-resolution generation without simply forcing the full 4K token grid through vanilla global self-attention.

- **Prompt-to-high-res restart:** when no input video is provided, the script first generates a normal 720p Wan2.2 latent, upsamples it in latent space, adds noise, and denoises at the requested high resolution.
- **Video-to-high-res restart:** when an input video is provided, the video can be resized in pixel space or encoded first and resized in latent space before the high-resolution denoising restart.
- **Block-tiled self-attention:** self-attention can be computed on spatial tiles inside each DiT block, then reassembled before cross-attention and FFN layers. This keeps text cross-attention operating on the full sequence.
- **Routed global context:** each tile can retrieve relevant same-frame global K/V grids from outside the tile, reducing the “independent tile” failure mode while avoiding full 4K attention cost.
- **Compressed-relative RoPE for global tokens:** routed global context is RoPE-encoded relative to the tile while keeping coordinates inside the Wan2.2 720p training range.
- **Joint local/global attention:** local tile tokens and routed global tokens can be concatenated into one attention computation, letting global context compete in the same softmax.

## Features ✨

- Prompt-only generation if `--video` is omitted
- Video input generation if `--video` is provided
- Pixel-space or latent-space input resizing via `--video_resize_mode`
- Partial high-resolution denoising with `--round_noise_steps`
- Wan-style sample shift control with `--sample_shift`
- Optional block-tiled self-attention
- Optional routed global self-attention
- Latent saving and decoded `.mp4` output
- Single-GPU and multi-GPU launch support

## Setup 🛠️

Place the Wan2.2 checkpoint directory where the script can find it, for example:

```text
Wan2.2-T2V-A14B/
```

Run commands from the repository root:

```bash
cd /home/gchen/CineScale
```

## Prompt-Only High-Resolution Generation 🧠

If `--video` is not provided, the script generates a 720p base latent from the prompt, upsamples that latent to `--size`, adds noise, and denoises.

```bash
torchrun --nproc_per_node=1 CineScale/Wan2.2/cinescale.py \
  --size "3840*2160" \
  --prompt "A cinematic shot of the Eiffel Tower framed by green foliage." \
  --ckpt_dir Wan2.2-T2V-A14B \
  --frame_num 1 \
  --round_noise_steps 5 \
  --sample_shift 12 \
  --save_latent CineScale/latent_result.pt \
  --save_video CineScale/result_video.mp4 \
  --offload_model true
```

## Video-Guided High-Resolution Restart 🎞️

Use `--video_resize_mode latent` to encode the input video first, then resize the latent to the requested output size.

```bash
torchrun --nproc_per_node=1 CineScale/Wan2.2/cinescale.py \
  --video CineScale/Eiffel_Tower.mp4 \
  --size "3840*2160" \
  --prompt "The video shows a view of the Eiffel Tower partially obscured by vibrant green foliage." \
  --ckpt_dir Wan2.2-T2V-A14B \
  --frame_num 1 \
  --video_resize_mode latent \
  --round_noise_steps 5 \
  --sample_shift 12 \
  --save_latent CineScale/latent_result.pt \
  --save_video CineScale/result_video.mp4 \
  --offload_model true
```

Use `--video_resize_mode pixel` if you want to resize the video in pixel space before VAE encoding.

## Tiled and Routed Attention 🧩

For larger outputs, block-tiled self-attention can reduce memory while keeping cross-attention and FFN layers on the reassembled full sequence.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --standalone --nproc_per_node=4 CineScale/Wan2.2/cinescale.py \
  --video CineScale/Eiffel_Tower.mp4 \
  --size "3840*2160" \
  --prompt "The video shows a view of the Eiffel Tower partially obscured by vibrant green foliage." \
  --ckpt_dir Wan2.2-T2V-A14B \
  --frame_num 41 \
  --video_resize_mode latent \
  --round_noise_steps 15 \
  --sample_shift 12 \
  --block_tiled_self_attn true \
  --block_tiled_self_attn_tile_height 24 \
  --block_tiled_self_attn_tile_width 24 \
  --block_tiled_self_attn_stride_height 20 \
  --block_tiled_self_attn_stride_width 20 \
  --block_tiled_self_attn_halo 6 \
  --block_tiled_self_attn_routed_topk 32 \
  --block_tiled_self_attn_routed_grid 2 \
  --block_tiled_self_attn_global_attention_mode joint \
  --block_tiled_self_attn_global_rope_threshold 40 \
  --block_tiled_self_attn_global_scale 1 \
  --save_latent CineScale/latent_result.pt \
  --save_video CineScale/result_video.mp4 \
  --ulysses_size 4 \
  --dit_fsdp \
  --t5_cpu \
  --offload_model true
```

## Important Options ⚙️

- `--size`: target output size, formatted as `"width*height"`.
- `--round_noise_steps`: how deep the restart goes. Larger values add more noise and allow more regeneration, but can drift farther from the input.
- `--sample_shift`: Wan sampling shift. Higher values push the schedule toward higher effective noise at the same step index.
- `--video_resize_mode latent`: encode first, then resize the latent.
- `--video_resize_mode pixel`: resize pixels first, then encode.
- `--block_tiled_self_attn true`: enable tiled self-attention inside DiT blocks.
- `--block_tiled_self_attn_halo`: extra context around each tile.
- `--block_tiled_self_attn_routed_topk`: number of routed global grids selected per frame.
- `--block_tiled_self_attn_routed_grid`: spatial grid size used for global routing chunks.
- `--block_tiled_self_attn_global_attention_mode joint`: concatenate local and routed global K/V into one softmax.
- `--block_tiled_self_attn_global_scale`: strength of routed global context.

## Notes ⚠️

- Larger `round_noise_steps` can create more detail, but it can also introduce tint, blur, or scene drift.
- Pure tiled self-attention can behave like independent local generation. Routed global K/V is intended to reduce that failure mode.
- Very large routed global scale can over-inject distant context and cause repeated objects.
- Latent-space resizing is often sharper than pixel-upsample-then-encode for this workflow, but it may introduce fine latent grain.

## Outputs 💾

- `--save_latent path.pt` writes the final latent.
- `--save_video path.mp4` decodes and saves the final video.
- `--decode_latent latent.pt --save_video out.mp4` decodes an existing latent without rerunning denoising.
