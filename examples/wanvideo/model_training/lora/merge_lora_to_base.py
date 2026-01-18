"""
Merge LoRA checkpoint into base Wan 2.1 I2V model weights.

Usage:
    python examples/wanvideo/model_training/lora/merge_lora_to_base.py \
        --lora_path ./models/train/Wan2.1-I2V-14B-480P_lora/epoch_0.safetensors \
        --output_path ./models/merged/Wan2.1-I2V-14B-480P-merged.safetensors \
        --alpha 1.0
"""

import torch
import argparse
from safetensors.torch import load_file, save_file
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.lora.general import GeneralLoRALoader


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to LoRA checkpoint (.safetensors or .pth)")
    parser.add_argument("--output_path", type=str, required=True, help="Output path for merged weights (.safetensors)")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA scaling factor (default: 1.0)")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-480P", help="Base model ID")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for merging")
    args = parser.parse_args()

    print(f"Loading base model: {args.model_id}")
    model_configs = [
        ModelConfig(model_id=args.model_id, origin_file_pattern="diffusion_pytorch_model*.safetensors"),
    ]
    tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/")

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=args.device,
        model_configs=model_configs,
        tokenizer_config=tokenizer_config,
    )

    print(f"Loading LoRA checkpoint: {args.lora_path}")
    if args.lora_path.endswith(".safetensors"):
        lora_state_dict = load_file(args.lora_path)
    else:
        lora_state_dict = torch.load(args.lora_path, map_location="cpu", weights_only=True)

    print(f"Fusing LoRA with alpha={args.alpha}")
    lora_loader = GeneralLoRALoader(device=args.device, torch_dtype=torch.bfloat16)
    lora_loader.fuse_lora_to_base_model(pipe.dit, lora_state_dict, alpha=args.alpha)

    print(f"Saving merged weights to: {args.output_path}")
    merged_state_dict = {k: v.to(torch.bfloat16).contiguous() for k, v in pipe.dit.state_dict().items()}
    save_file(merged_state_dict, args.output_path)

    print("Done!")


if __name__ == "__main__":
    main()
