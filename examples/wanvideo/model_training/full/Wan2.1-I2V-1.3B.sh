export DIFFSYNTH_SKIP_DOWNLOAD=true

accelerate launch --num_processes 8 examples/wanvideo/model_training/train.py \
  --task i2v \
  --dataset_base_path train_subset \
  --dataset_metadata_path train_subset/train_subset.csv \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --dataset_repeat 100 \
  --dataset_num_workers 4 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --gradient_accumulation_steps 4 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.1-I2V-1.3B_full" \
  --save_steps 500 \
  --trainable_models "dit"
