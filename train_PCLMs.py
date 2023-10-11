
import torch
from modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer, GPT2Config
from transformers import get_linear_schedule_with_warmup, AdamW
import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
import argparse
import random
import numpy as np
from tqdm import tqdm, trange
import json
import math
import os
import shutil
import pdb

# torch.autograd.set_detect_anomaly(True)

class tokendataset(Dataset):
    def __init__(self, dataset_path):
        self.file_path = dataset_path
        dataset = []
        file_row = 0
        with open(self.file_path, 'r') as f:
            for line in f.readlines():
                file_row += 1
                dataset.append(json.loads(line))
        self.file_row = file_row
        self.dataset = dataset

    def __len__(self):
        return self.file_row
    
    def __getitem__(self, idx):
        return self.dataset[idx]

def padding_fuse_fn(data_list):
    input_ids = []
    attention_masks = []
    sentiment = []
    text_length = []
    for item in data_list:
        text_length.append(len(item['text']))
        sentiment.append([item['label']])
    max_text_len = max(text_length)
    for i, item in enumerate(data_list):
        text_pad_len = max_text_len - text_length[i]

        attention_mask = [1] * text_length[i] + [0] * text_pad_len
        text = item["text"] + [50256] * text_pad_len

        input_ids.append(text)
        attention_masks.append(attention_mask)
    
    batch = {}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_masks
    batch["sentiment"] = sentiment

    return batch

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def train(args):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    set_seed(args)

    model_config = GPT2Config.from_pretrained(args.model_name_or_path)
    model_config.prefix_len = args.prefix_len
    model_config.prefix_mid_size = args.prefix_mid_size
    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path, config=model_config)
    for param in model.named_parameters():
        if 'prefix' in param[0]:
            continue
        else:
            param[1].requires_grad = False
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    args.tokenizer = tokenizer
    model.to(args.device)

    dataset_list = [args.pos_dataset_path, args.neg_dataset_path, args.world_dataset_path, args.sports_dataset_path, args.business_dataset_path, args.science_dataset_path, args.nontoxic_dataset_path, args.toxic_dataset_path]

    for i, dataset_path in enumerate(dataset_list):
        print("the current training dataset is: {}".format(dataset_path))
        print("="*100)
        train_dataset = tokendataset(dataset_path)
        file_row = train_dataset.file_row
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=train_sampler)     
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        num_train_steps = math.floor(len(train_dataset) / (
                args.batch_size * args.gradient_accumulation_steps)) * args.num_train_epochs
        num_warmup_steps = math.floor(num_train_steps * args.warmup_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.train()
        model.zero_grad()
        loss_fct = CrossEntropyLoss()
        args.loss_fct = loss_fct
        logger.info('start_training')

        current_epoch = 0
        for epoch in trange(int(args.num_train_epochs), desc='Epoch'):
            current_epoch += 1
            for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
                global_step += 1

                input_ids, attention_mask, sentiment = batch['input_ids'], batch['attention_mask'], batch['sentiment']
                eos_token_ids = torch.tensor(tokenizer.encode(tokenizer.eos_token))
                eos_token_ids = eos_token_ids.expand(args.batch_size, eos_token_ids.shape[0])
                input_ids = torch.tensor(input_ids)
                input_ids = torch.cat([eos_token_ids, input_ids], dim=1).to(args.device)
                eos_token_mask = torch.tensor([1]).expand(args.batch_size, 1)
                prefix_mask = torch.tensor([1] * args.prefix_len).expand(args.batch_size, args.prefix_len)
                attention_mask = torch.tensor(attention_mask)
                attention_mask = torch.cat([eos_token_mask, attention_mask], dim=1)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1).to(args.device)
                sentiment = torch.tensor(sentiment).to(args.device)
                # pdb.set_trace()
                dic = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, use_cache=False, use_prefix=True, prefix_id=i)
                logits = dic.logits
                shift_logits = logits[:, :-1, :].contiguous()
                labels = input_ids[:, 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
                loss.backward()
                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    logs['epoch'] = current_epoch
                    logs['step'] = global_step
                    logs['loss'] = loss_scalar
                    logging_loss = tr_loss
                    print(logs)
        
        if current_epoch == args.num_train_epochs:
            output_dir = os.path.join(args.output_dir, 'ckpt-{}-prefixlen-{}-bs-{}-epoch-{}'.format(i, args.prefix_len, args.batch_size * args.gradient_accumulation_steps, current_epoch))
            model_to_save = (model.module if hasattr(model, 'module') else model)
            model_to_save.save_pretrained(output_dir)
            model_config.save_pretrained(output_dir)

    logger.info(' global_step = %s, average loss = %s', global_step, tr_loss / global_step)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_dataset_path", default="../dataset/sentiment-imdb/imdb_5k_pos_tokenized.json", type=str)
    parser.add_argument("--neg_dataset_path", default="../dataset/sentiment-imdb/imdb_5k_neg_tokenized.json", type=str)
    parser.add_argument("--world_dataset_path", default="../dataset/topic-agnews/agnews_5k_world_tokenized.json", type=str)
    parser.add_argument("--sports_dataset_path", default="../dataset/topic-agnews/agnews_5k_sports_tokenized.json", type=str)
    parser.add_argument("--business_dataset_path", default="../dataset/topic-agnews/agnews_5k_business_tokenized.json", type=str)
    parser.add_argument("--science_dataset_path", default="../dataset/topic-agnews/agnews_5k_science_tokenized.json", type=str)
    parser.add_argument("--nontoxic_dataset_path", default="../dataset/detoxification-jigsaw/jigsaw_5k_nontoxic_tokenized.json", type=str)
    parser.add_argument("--toxic_dataset_path", default="../dataset/detoxification-jigsaw/jigsaw_5k_toxic_tokenized.json", type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--prefix_len", default=20, type=int)
    parser.add_argument("--prefix_mid_size", default=512, type=int)
    parser.add_argument("--output_dir", default=None, type=str)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--learning_rate", default=3e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=5, type=int)
    parser.add_argument("--warmup_rate", default=0.1, type=float)
    parser.add_argument("--logging_steps", default=50, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--save_steps", default=40000, type=int)
    parser.add_argument("--device_num", default=None, type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda:{}".format(args.device_num))
    
    train(args)
