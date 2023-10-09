model_name_or_path=../models/ckpt_for_sentiment_and_topic
samples=50
task_mode=sentiment
lambda_cs=140.0
device_num=3

python ../air_decoding.py --model_name_or_path $model_name_or_path --samples $samples --task_mode $task_mode --lambda_cs $lambda_cs --device_num $device_num
