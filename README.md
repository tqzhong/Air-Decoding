# Air-Decoding
This repository contains code for paper Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time Controllable Text Generation which has been accepted to appear at EMNLP 2023. If you have any questions, please feel free to create an issue or contact the email: ztq602656097@mail.ustc.edu.cn

## Contents

```
├── dataset
│   ├── detoxification-jigsaw
│   ├── sentiment-imdb
│   └── topic-agnews
├── models
│   ├── best_sentiment_classifier
│   ├── best_topic_classifier
│   ├── ckpt_for_detoxification
│   └── ckpt_for_sentiment_and_topic
├── scripts
├── test_data
```

## Description of Main files

- `air-decoding.py`: Air-Decoding algorithm implementation for generating text with desired attributes
- `train_PCLMs.py`: Training PC-LMs with desired attributes
- `eval_sent_acc.py`: evaluate the sentiment accuracy of generated text
- `eval_topic_acc.py`: evaluate the topic accuracy of generated text
- `eval_toxic`: evaluate the average toxicity of generated text
- `eval_perplexity.py`: evaluate the average perplexity of generated text
- `eval_dist.py`: evaluate the dist-1, dist-2, dist-3 of generated text
- `/scripts`: it contains the bash commands for model training, controllable text generation, and evaluation

## Experiment Setup

- Install the following environment

  ```shell
  pip install -r requirements.txt
  ```

- Download the models: [click here](https://drive.google.com/file/d/1Su5-QT2nIjjZ_pcyGkc5f-AR6vOs0ZVw/view?usp=sharing)

- After downloading, you will get the "models.zip" file, and you should move it to the main directory.

  ```shell
  unzip models.zip
  rm models.zip
  ```

## Training PC-LMs

It contains the training process of PC-LMs.

#### Training Implementation

```shell
mkdir ckpt
cd ./scripts
bash train_PCLMs.sh
```

#### Parameter Configuration

`--model_name_or_path`: pretrained language model path, i.e., GPT2-medium or GPT2-large

`--prefix_len`: the length of prefix

`--prefix_mid_size`: the dimension of reparameterization in prefix-tuning

`--output_dir`: the save path for the output model

## Generating controllable text using Air-Decoding

It contains the generation process of Air-Decoding.

#### Generation Implementation

```shell
cd ./scripts
bash generate_sentiment.sh
bash generate_topic.sh
bash generate_detoxification.sh
```

#### Parameter Configuration

- `--model_name_or_path`: fine-tuned PC-LMs model path
- `--length`: the length of generated text
- `--samples`: the number of generated texts for each prompt
- `--lambda_cs`: control strength

## Evaluation

It contains the evaluation process.

#### Evaluation Implementation

```shell
cd ./scripts
bash eval_sent_acc.sh
bash eval_topic_acc.sh
bash eval_toxic.sh
bash eval_perplexity.sh
bash eval_dist.sh
```

#### Parameter Configuration

- `--model_name_or_path`: fine-tuned classifier model for sentiment or topic evaluation and GPT2-large model for perplexity evaluation
- `--dataset_path`: the path of the file under test, which is a JSONL file. Each data entry in the file includes a 'text' field and its corresponding attribute label