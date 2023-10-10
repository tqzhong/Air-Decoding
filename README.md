# Air-Decoding
This repository contains code for paper Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time Controllable Text Generation which has been accepted to appear at EMNLP 2023. If you have any questions, please feel free to create an issue or contact the email: ztq602656097@mail.ustc.edu.cn

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

## Training PC-LMs

It contains the training process of PC-LMs.

```shell
mkdir ckpt
cd ./scripts
bash train_PCLMs.sh
```

## Generating controllable text using Air-Decoding

It contains the generation process of Air-Decoding.

```shell
cd ./scripts
bash generate_sentiment.sh
bash generate_topic.sh
bash generate_detoxification.sh
```

## Evaluation

It contains the evaluation process.

```shell
cd ./scripts
bash eval_sent_acc.sh
bash eval_topic_acc.sh
bash eval_toxic.sh
bash eval_perplexity.sh
```

