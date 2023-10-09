model_name_or_path=../models/ckpt_for_detoxification
samples=20
task_mode=detoxification
lambda_cs=120.0
device_num=3

python ../air_decoding.py --model_name_or_path $model_name_or_path --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num