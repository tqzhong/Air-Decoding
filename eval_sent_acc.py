
from model_sent import RobertaForPreTraining
from transformers import RobertaTokenizer
# from train import tokendataset, padding_fuse_fn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
# from evaluate import load
import argparse
import torch
import json
import pdb

# DEVICE = torch.device("cuda:1")
MAXLEN = 512
BATCH_SIZE = 4

class tokendataset(Dataset):
    def __init__(self, dataset_path):
        file_row_pos = 0
        file_row_neg = 0
        for item in dataset_path:
            sentiment = item['sentiment']
            if sentiment == 0:
                file_row_neg += 1
            elif sentiment == 1:
                file_row_pos += 1
            else:
                raise Exception("Sentiment Type Error!")
            
        self.file_row = len(dataset_path)
        self.file_row_pos = file_row_pos
        self.file_row_neg = file_row_neg
        self.dataset = dataset_path

    def __len__(self):
        return self.file_row

    def __getitem__(self, idx):
        return self.dataset[idx]

def padding_fuse_fn(data_list):
    input_ids = []
    attention_masks = []
    sentiment = []
    text_length = []
    ppl = []
    for item in data_list:
        text_length.append(len(item["text"]))
        sentiment.append([item["sentiment"]])
    max_text_len = max(text_length)

    for i, item in enumerate(data_list):
        text_pad_len = max_text_len - text_length[i]

        attention_mask = [1] * text_length[i] + [0] * text_pad_len
        text = item["text"] + [0] * text_pad_len

        input_ids.append(text)
        attention_masks.append(attention_mask)
    
    batch = {}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_masks
    batch["sentiment"] = sentiment

    return batch

def tokenized(dataset_path=None, tokenizer=None):

    output_data = list()

    f = open(dataset_path, 'r')
    for line in f.readlines():
        data = {}
        dic = json.loads(line)
        if dic['sentiment'] == 'Negative':
            data['sentiment'] = 0
        elif dic['sentiment'] == 'Positive':
            data['sentiment'] = 1
        # if 'label' in dic:
        #     if dic['label'] == 'Positive' or dic['label'] == 'positive' or dic['label'] == 1 :
        #         data['sentiment'] = 1
        #     elif dic['label'] == 'Negative' or dic['label'] == 'negative' or dic['label'] == 0:
        #         data['sentiment'] = 0
        #     else:
        #         raise Exception("Wrong label value!")
        # elif 'sentiment' in dic:
        #     if dic['sentiment'] == 0 or dic['sentiment'] == '0' or dic['sentiment'] == 'Negative':
        #         data['sentiment'] = 0
        #     elif dic['sentiment'] == 1 or dic['sentiment'] == '1' or dic['sentiment'] == 'Positive':
        #         data['sentiment'] = 1
        #     else:
        #         raise Exception("Wrong sentiment value")
        # elif 'sent' in dic:
        #     if dic['sent'] == 'neg':
        #         data['sentiment'] = 0
        #     elif dic['sent'] == 'pos':
        #         data['sentiment'] = 1
        #     else:
        #         raise Exception("Wrong sent value")
        # else:
        #     raise Exception("Wrong key value")
        if 'text' in dic:
            data['text'] = tokenizer.encode(dic['text'], max_length=MAXLEN, truncation=True)
        elif 'review' in dic:
            data['text'] = tokenizer.encode(dic['review'], max_length=MAXLEN, truncation=True)

        output_data.append(data)
    
    f.close()

    return output_data

        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--device_num", default=None, type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda:{}".format(args.device_num))

    # tokenized the data in dataset_path
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    output_data = tokenized(dataset_path=args.dataset_path, tokenizer=tokenizer)
    args.output_data = output_data
    # pdb.set_trace()

    test_dataset = tokendataset(args.output_data)
    file_row = test_dataset.file_row
    file_row_pos = test_dataset.file_row_pos
    file_row_neg = test_dataset.file_row_neg
    test_sampler = torch.utils.data.RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=test_sampler)

    model = RobertaForPreTraining.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    tp_all, tp0_all, tp1_all, fp0_all, fp1_all = 0, 0, 0, 0, 0
    tr_loss = 0.0
    logs = {}

    for step, batch in enumerate(test_dataloader):
        input_ids, attention_mask, sentiment = batch['input_ids'], batch['attention_mask'], batch['sentiment']

        input_ids = torch.tensor(input_ids).to(args.device)
        attention_mask = torch.tensor(attention_mask).to(args.device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        sentiment = torch.tensor(sentiment).to(args.device)

        loss, tp, tp0, tp1, fp0, fp1 = model(input_ids=input_ids, attention_mask=attention_mask, sentiment=sentiment)
        tp_all += tp
        tp0_all += tp0
        tp1_all += tp1
        fp0_all += fp0
        fp1_all += fp1

        tr_loss += loss.item()

    acc = tp_all / (tp_all + fp0_all + fp1_all)
    acc0 = tp0_all / (tp0_all + fp0_all)
    acc1 = tp1_all / (tp1_all + fp1_all)
    logs['acc'] = float('{:.4f}'.format(acc))
    logs['acc0'] = float('{:.4f}'.format(acc0))
    logs['acc1'] = float('{:.4f}'.format(acc1))
    logs['total_loss'] = tr_loss
    # compute average ppl
    # f = open(args.dataset_path, 'r')
    # text_list = []
    # for line in f.readlines():
    #     dic = json.loads(line)
    #     text = dic['text']
    #     text_list.append(text)
    # # print(text_list)
    # perplexity = load("perplexity", module_type="metric")
    # results = perplexity.compute(model_id='gpt2', predictions=text_list)
    # logs['avg_ppl'] = round(results["mean_perplexity"], 2)
    # print('\n\n')
    print(logs)
    
if __name__ == "__main__":
    main()