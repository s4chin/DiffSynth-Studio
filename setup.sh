# env setup
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Download data
aws s3 cp ... ./
unzip train_subset.zip

sed -i '1s/video_path,image_path/video,input_image/' train_subset/train_subset.csv && head -3 /mnt/localssd/DiffSynth-Studio/train_subset/train_subset.csv
sed -i 's|train_subset/|train_subset_videos/|g' train_subset/train_subset.csv

bash examples/wanvideo/model_training/lora/Wan2.1-I2V-14B-480P-8gpu.sh


