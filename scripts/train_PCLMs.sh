model_name_or_path=gpt2-medium
device_num=3
output_dir=./ckpt

python ../train_PCLMs.py --model_name_or_path $model_name_or_path --output_dir $output_dir --device_num $device_num