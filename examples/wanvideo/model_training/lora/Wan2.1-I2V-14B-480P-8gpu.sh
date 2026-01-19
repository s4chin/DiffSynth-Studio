# export DIFFSYNTH_SKIP_DOWNLOAD=true

accelerate launch --num_processes 8 examples/wanvideo/model_training/train.py \
  --dataset_base_path train_subset \
  --dataset_metadata_path train_subset/train_subset.csv \
  --data_file_keys "video,input_image" \
  --height 480 \
  --width 832 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-I2V-14B-480P:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-I2V-14B-480P:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
  --learning_rate 1e-4 \
  --num_epochs 1 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-14B-480P_lora" \
  --save_steps 5000 \
  --lora_base_model "dit" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "input_image"
