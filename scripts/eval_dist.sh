dataset_path=../test_data/Air_sentiment_140.0.jsonl
model_name_or_path=roberta-large

python ../eval_dist.py --dataset_path $dataset_path --model_name_or_path $model_name_or_path