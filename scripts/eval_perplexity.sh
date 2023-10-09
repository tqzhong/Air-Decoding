model_name_or_path=gpt2-large
dataset_path=../test_data/Air_sentiment_140.0.jsonl
device_num=3

python ../eval_perplexity.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --device_num $device_num