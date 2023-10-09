model_name_or_path=../models/best_topic_classifier
device_num=3
dataset_path=../test_data/Air_topic_60.0.jsonl

python ../eval_topic_acc.py --dataset_path $dataset_path --model_name_or_path $model_name_or_path --device_num $device_num