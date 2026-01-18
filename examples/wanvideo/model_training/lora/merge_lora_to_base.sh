python examples/wanvideo/model_training/lora/merge_lora_to_base.py \
  --lora_path ./models/train/Wan2.1-I2V-14B-480P_lora/epoch_0.safetensors \
  --output_path ./models/merged/Wan2.1-I2V-14B-480P-merged.safetensors \
  --alpha 1.0 \
  --model_id "Wan-AI/Wan2.1-I2V-14B-480P" \
  --device cuda
