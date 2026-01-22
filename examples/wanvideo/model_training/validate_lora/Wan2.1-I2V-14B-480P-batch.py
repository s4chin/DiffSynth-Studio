"""
Batch validation script for Wan2.1-I2V-14B-480P LoRA model.
Reads a val.csv file and generates videos for all entries.

Supports two workflows:
1. Single image conditioning: Standard I2V with one input image
2. Dual image conditioning: Start with image1, switch to image2 after n steps
"""
import torch
import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from diffsynth.utils.data import save_video, VideoData
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, model_fn_wan_video


# Fixed parameters
HEIGHT = 480
WIDTH = 832
FPS = 16
LORA_ALPHA = 1.0
NEGATIVE_PROMPT = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"


def parse_args():
    parser = argparse.ArgumentParser(description="Batch validation for Wan2.1-I2V-14B-480P LoRA")
    parser.add_argument(
        "--val_csv",
        type=str,
        required=True,
        help="Path to the validation CSV file (same format as train.csv)"
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="",
        help="Base path prefix for video/image paths in CSV (if paths are relative)"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to the LoRA checkpoint"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Random seed for generation"
    )
    parser.add_argument(
        "--video_column",
        type=str,
        default="video",
        help="Column name for video paths in CSV"
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="input_image",
        help="Column name for input image paths in CSV"
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Column name for prompts in CSV"
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation if output video already exists"
    )
    # Dual image conditioning arguments
    parser.add_argument(
        "--image1_column",
        type=str,
        default=None,
        help="Column name for first conditioning image (for dual-image mode)"
    )
    parser.add_argument(
        "--image2_column",
        type=str,
        default=None,
        help="Column name for second conditioning image (for dual-image mode)"
    )
    parser.add_argument(
        "--switch_step",
        type=int,
        default=None,
        help="Step at which to switch from image1 to image2 conditioning (0-indexed). If not set, uses single-image mode."
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    return parser.parse_args()


def get_output_dir_from_lora_path(lora_path):
    """Generate output directory name from LoRA checkpoint name."""
    # Extract checkpoint name without extension (e.g., "epoch-4" from "path/to/epoch-4.safetensors")
    checkpoint_name = Path(lora_path).stem
    return f"output_videos/output_{checkpoint_name}"


def load_pipeline(lora_path):
    """Load the WanVideo pipeline with LoRA weights."""
    print("Loading pipeline...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="diffusion_pytorch_model*.safetensors"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="Wan2.1_VAE.pth"),
            ModelConfig(model_id="Wan-AI/Wan2.1-I2V-14B-480P", origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        ],
    )
    
    if lora_path and os.path.exists(lora_path):
        print(f"Loading LoRA from: {lora_path}")
        pipe.load_lora(pipe.dit, lora_path, alpha=LORA_ALPHA)
    else:
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    
    return pipe


def get_output_path(video_path, output_dir):
    """Generate output path keeping the original video filename."""
    video_name = Path(video_path).name
    return os.path.join(output_dir, video_name)


def encode_image_clip(pipe, image, height, width):
    """Encode image using CLIP encoder."""
    if image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
        return None
    pipe.load_models_to_device(["image_encoder"])
    img = pipe.preprocess_image(image.resize((width, height))).to(pipe.device)
    clip_context = pipe.image_encoder.encode_image([img])
    clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
    return clip_context


def encode_image_vae(pipe, image, num_frames, height, width, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
    """Encode image using VAE encoder for I2V conditioning."""
    if image is None or not pipe.dit.require_vae_embedding:
        return None
    pipe.load_models_to_device(["vae"])
    img = pipe.preprocess_image(image.resize((width, height))).to(pipe.device)
    msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
    msk[:, 1:] = 0
    vae_input = torch.concat([img.transpose(0, 1), torch.zeros(3, num_frames - 1, height, width).to(img.device)], dim=1)
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
    msk = msk.transpose(1, 2)[0]
    y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
    y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
    y = torch.concat([msk, y])
    y = y.unsqueeze(0)
    y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
    return y


def encode_prompt(pipe, prompt):
    """Encode text prompt."""
    pipe.load_models_to_device(["text_encoder"])
    ids, mask = pipe.tokenizer(prompt, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    seq_lens = mask.gt(0).sum(dim=1).long()
    prompt_emb = pipe.text_encoder(ids, mask)
    for i, v in enumerate(seq_lens):
        prompt_emb[:, v:] = 0
    return prompt_emb


@torch.no_grad()
def generate_with_dual_conditioning(
    pipe,
    prompt,
    negative_prompt,
    image1,
    image2,
    switch_step,
    seed=1,
    height=HEIGHT,
    width=WIDTH,
    num_frames=81,
    num_inference_steps=50,
    cfg_scale=5.0,
    sigma_shift=5.0,
    tiled=True,
    tile_size=(30, 52),
    tile_stride=(15, 26),
    progress_bar_cmd=tqdm,
):
    """
    Generate video with dual image conditioning.
    Starts with image1 conditioning, switches to image2 after switch_step.
    """
    # Check and resize dimensions
    height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
    
    # Initialize scheduler
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)
    
    # Initialize noise
    length = (num_frames - 1) // 4 + 1
    shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
    noise = pipe.generate_noise(shape, seed=seed, rand_device="cpu")
    latents = noise.clone()
    
    # Encode prompts
    context_posi = encode_prompt(pipe, prompt)
    context_nega = encode_prompt(pipe, negative_prompt)
    
    # Encode both images with CLIP
    clip_feature_1 = encode_image_clip(pipe, image1, height, width)
    clip_feature_2 = encode_image_clip(pipe, image2, height, width)
    
    # Encode both images with VAE
    y_1 = encode_image_vae(pipe, image1, num_frames, height, width, tiled, tile_size, tile_stride)
    y_2 = encode_image_vae(pipe, image2, num_frames, height, width, tiled, tile_size, tile_stride)
    
    # Prepare inputs shared between positive and negative
    inputs_shared = {
        "latents": latents,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "cfg_merge": False,
    }
    
    # Denoise
    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    
    for progress_id, timestep in enumerate(progress_bar_cmd(pipe.scheduler.timesteps, desc="Denoising")):
        # Select conditioning based on current step
        if progress_id < switch_step:
            clip_feature = clip_feature_1
            y = y_1
        else:
            clip_feature = clip_feature_2
            y = y_2
        
        # Timestep
        timestep_tensor = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # Positive inference
        noise_pred_posi = model_fn_wan_video(
            **models,
            latents=inputs_shared["latents"],
            timestep=timestep_tensor,
            context=context_posi,
            clip_feature=clip_feature,
            y=y,
            cfg_merge=False,
        )
        
        # Negative inference for CFG
        if cfg_scale != 1.0:
            noise_pred_nega = model_fn_wan_video(
                **models,
                latents=inputs_shared["latents"],
                timestep=timestep_tensor,
                context=context_nega,
                clip_feature=clip_feature,
                y=y,
                cfg_merge=False,
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi
        
        # Scheduler step
        inputs_shared["latents"] = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"])
    
    # Decode
    pipe.load_models_to_device(['vae'])
    video = pipe.vae.decode(inputs_shared["latents"], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    
    return video


def main():
    args = parse_args()
    
    # Determine mode: dual-image or single-image
    dual_image_mode = (
        args.image1_column is not None 
        and args.image2_column is not None 
        and args.switch_step is not None
    )
    
    if dual_image_mode:
        print("=" * 50)
        print("DUAL-IMAGE CONDITIONING MODE")
        print(f"  Image 1 column: {args.image1_column}")
        print(f"  Image 2 column: {args.image2_column}")
        print(f"  Switch step: {args.switch_step} / {args.num_inference_steps}")
        print("=" * 50)
    else:
        print("=" * 50)
        print("SINGLE-IMAGE CONDITIONING MODE")
        print("=" * 50)
    
    # Generate output directory from LoRA checkpoint name
    output_dir = get_output_dir_from_lora_path(args.lora_path)
    if dual_image_mode:
        output_dir = f"{output_dir}_switch{args.switch_step}"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load validation CSV
    print(f"Loading validation CSV: {args.val_csv}")
    df = pd.read_csv(args.val_csv)
    print(f"Found {len(df)} entries in validation set")
    
    # Validate required columns
    required_columns = [args.video_column, args.prompt_column]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV. Available columns: {list(df.columns)}")
    
    # Validate columns based on mode
    has_image_column = False  # Default for scoping
    if dual_image_mode:
        for col in [args.image1_column, args.image2_column]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV for dual-image mode. Available columns: {list(df.columns)}")
        if args.switch_step < 0 or args.switch_step >= args.num_inference_steps:
            raise ValueError(f"switch_step must be between 0 and {args.num_inference_steps - 1}, got {args.switch_step}")
    else:
        # Check for image column (optional - can use first frame of video)
        has_image_column = args.image_column in df.columns
        if not has_image_column:
            print(f"Note: Image column '{args.image_column}' not found. Will use first frame of video as input.")
    
    # Load pipeline
    pipe = load_pipeline(args.lora_path)
    
    # Process each entry
    successful = 0
    failed = 0
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating videos"):
        video_path = row[args.video_column]
        prompt = row[args.prompt_column]
        
        # Handle relative paths
        if args.base_path:
            video_path = os.path.join(args.base_path, video_path)
        
        # Get output path (same name as input video)
        output_path = get_output_path(row[args.video_column], output_dir)
        
        # Skip if already exists
        if args.skip_existing and os.path.exists(output_path):
            print(f"Skipping (exists): {output_path}")
            skipped += 1
            continue
        
        try:
            if dual_image_mode:
                # DUAL-IMAGE MODE
                image1_path = row[args.image1_column]
                image2_path = row[args.image2_column]
                if args.base_path:
                    image1_path = os.path.join(args.base_path, image1_path)
                    image2_path = os.path.join(args.base_path, image2_path)
                
                image1 = Image.open(image1_path).convert("RGB")
                image2 = Image.open(image2_path).convert("RGB")
                # Resize to target dimensions
                image1 = image1.resize((WIDTH, HEIGHT), Image.LANCZOS)
                image2 = image2.resize((WIDTH, HEIGHT), Image.LANCZOS)
                
                # Generate video with dual conditioning
                video = generate_with_dual_conditioning(
                    pipe=pipe,
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    image1=image1,
                    image2=image2,
                    switch_step=args.switch_step,
                    seed=args.seed+idx,
                    height=HEIGHT,
                    width=WIDTH,
                    num_inference_steps=args.num_inference_steps,
                    tiled=True,
                )
            else:
                # SINGLE-IMAGE MODE
                if has_image_column:
                    image_path = row[args.image_column]
                    if args.base_path:
                        image_path = os.path.join(args.base_path, image_path)
                    input_image = Image.open(image_path).convert("RGB")
                    # Resize to target dimensions
                    input_image = input_image.resize((WIDTH, HEIGHT), Image.LANCZOS)
                else:
                    # Use first frame of video
                    input_image = VideoData(video_path, height=HEIGHT, width=WIDTH)[0]
                
                # Generate video
                video = pipe(
                    prompt=prompt,
                    negative_prompt=NEGATIVE_PROMPT,
                    input_image=input_image,
                    seed=args.seed+idx,
                    num_inference_steps=args.num_inference_steps,
                    tiled=True
                )
            
            # Save video
            save_video(video, output_path, fps=FPS, quality=5)
            print(f"Saved: {output_path}")
            successful += 1
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue
    
    # Print summary
    print("\n" + "=" * 50)
    print("Validation Complete!")
    print(f"  Mode: {'Dual-image' if dual_image_mode else 'Single-image'}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(df)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
