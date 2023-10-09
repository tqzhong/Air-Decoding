from model_topic import RobertaForPreTraining
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
import torch
import json

MAXLEN = 512
BATCH_SIZE = 32

class tokendataset(Dataset):
    def __init__(self, dataset_path):
        file_row_world = 0
        file_row_sports = 0
        file_row_business = 0
        file_row_science = 0
        for item in dataset_path:
            topic = item['topic']
            if topic == 0:
                file_row_world += 1
            elif topic == 1:
                file_row_sports += 1
            elif topic == 2:
                file_row_business += 1
            elif topic == 3:
                file_row_science += 1
            else:
                raise Exception("Topic Type Error!")

        self.file_row = len(dataset_path)
        self.file_row_world = file_row_world
        self.file_row_sports = file_row_sports
        self.file_row_business = file_row_business
        self.file_row_science = file_row_science
        self.dataset = dataset_path

    def __len__(self):
        return self.file_row

    def __getitem__(self, idx):
        return self.dataset[idx]
    
def padding_fuse_fn(data_list):
    input_ids = []
    attention_masks = []
    topic = []
    text_length = []
    for item in data_list:
        text_length.append(len(item["text"]))
        topic.append([item["topic"]])
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
    batch["topic"] = topic
    return batch

def tokenized(dataset_path=None, tokenizer=None):

    output_data = list()

    f = open(dataset_path, 'r')
    for line in f.readlines():
        data = {}
        dic = json.loads(line)
        if 'label' in dic:
            if dic['label'] == 'World' or dic['label'] == 'world' or dic['label'] == 0:
                data['topic'] = 0
            elif dic['label'] == 'Sports' or dic['label'] == 'sports' or dic['label'] == 1:
                data['topic'] = 1
            elif dic['label'] == 'Business' or dic['label'] == 'business' or dic['label'] == 2:
                data['topic'] = 2
            elif dic['label'] == 'Science' or dic['label'] == 'science' or dic['label'] == 3:
                data['topic'] = 3
            else:
                raise Exception("Wrong label value!")
        elif 'topic' in dic:
            if dic['topic'] == 0 or dic['topic'] == '0' or dic['topic'] == 'World':
                data['topic'] = 0
            elif dic['topic'] == 1 or dic['topic'] == '1' or dic['topic'] == 'Sports':
                data['topic'] =1
            elif dic['topic'] == 2 or dic['topic'] == '2' or dic['topic'] == 'Business':
                data['topic'] = 2
            elif dic['topic'] ==3 or dic['topic'] == '3' or dic['topic'] == 'Science':
                data['topic'] = 3
            else:
                raise Exception("Wrong topic value")
        else:
            raise Exception("Wrong key value")
        data['text'] = tokenizer.encode(dic['text'], max_length=MAXLEN, truncation=True)

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

    # tokenized the data in dataset_path, original roberta tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    output_data = tokenized(dataset_path=args.dataset_path, tokenizer=tokenizer)
    args.output_data = output_data

    test_dataset = tokendataset(args.output_data)
    file_row = test_dataset.file_row
    file_row_world = test_dataset.file_row_world
    file_row_sports = test_dataset.file_row_sports
    file_row_business = test_dataset.file_row_business
    file_row_science = test_dataset.file_row_science
    test_sampler = torch.utils.data.RandomSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=False, collate_fn=padding_fuse_fn, sampler=test_sampler)

    model = RobertaForPreTraining.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    tp_all, tp0_all, tp1_all, tp2_all, tp3_all = 0, 0, 0, 0, 0
    fp0_all, fp1_all, fp2_all, fp3_all = 0, 0, 0, 0
    tr_loss = 0.0
    logs = {}

    for step, batch in enumerate(test_dataloader):
        input_ids, attention_mask, topic = batch['input_ids'], batch['attention_mask'], batch['topic']

        input_ids = torch.tensor(input_ids).to(args.device)
        attention_mask = torch.tensor(attention_mask).to(args.device)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        topic = torch.tensor(topic).to(args.device)

        loss, tp, tp0, tp1, tp2, tp3, fp0, fp1, fp2, fp3 = model(input_ids=input_ids, attention_mask=attention_mask, topic=topic)
        tp_all += tp
        tp0_all += tp0
        tp1_all += tp1
        tp2_all += tp2
        tp3_all += tp3
        fp0_all += fp0
        fp1_all += fp1
        fp2_all += fp2
        fp3_all += fp3
        tr_loss += loss.item()

    acc = tp_all / (tp_all + fp0_all + fp1_all + fp2_all + fp3_all)
    acc0 = tp0_all / (tp0_all + fp0_all)
    acc1 = tp1_all / (tp1_all + fp1_all)
    acc2 = tp2_all / (tp2_all + fp2_all)
    acc3 = tp3_all / (tp3_all + fp3_all)
    logs['acc'] = float('{:.4f}'.format(acc))
    logs['acc0'] = float('{:.4f}'.format(acc0))
    logs['acc1'] = float('{:.4f}'.format(acc1))
    logs['acc2'] = float('{:.4f}'.format(acc2))
    logs['acc3'] = float('{:.4f}'.format(acc3))
    logs['total_loss'] = tr_loss
    print(logs)

if __name__ == "__main__":
    main()
