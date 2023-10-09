import argparse
import json
from tqdm import tqdm
import pdb
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

TEXT_LENGTH = 50

def cal_ppl(text=None, args=None):
    input_text = torch.tensor([tokenizer(tokenizer.eos_token + text).input_ids]).to(args.device)
    output = args.model(input_ids=input_text, return_dict=True, use_cache=True)
    logits = output.logits
    shift_logits = logits[:, :-1, :].squeeze()
    shift_logits = torch.softmax(shift_logits, dim=-1)
    index = torch.tensor(tokenizer(text).input_ids).to(args.device)
    probs = []
    for i in range(shift_logits.shape[0]):
        prob = torch.index_select(shift_logits[i], -1, index[i]).item()
        probs.append(prob)
    if 0 in probs:
        return -1
    ppl = 1
    for prob in probs:
        ppl *= (1 / prob) ** (1 / shift_logits.shape[0])
    return ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--device_num", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    args = parser.parse_args()
    args.device = torch.device("cuda:{}".format(args.device_num))

    model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()
    args.tokenizer = tokenizer
    args.model = model

    ppl_list = []
    f = open(args.dataset_path, 'r')
    for line in tqdm(f.readlines()):
        dic = json.loads(line)
        text = dic['text']
        if isinstance(text, dict):
            for i in range(len(text)):
                text_i = text[str(i)]
                if len(text_i.split(' ')) < 10:
                    continue
                ppl = cal_ppl(text=text_i, args=args)
                if ppl == -1:
                    continue
                ppl_list.append(ppl)
        elif isinstance(text, str):
            if len(text.split(' ')) < 10:
                continue
            ppl = cal_ppl(text=text, args=args)
            if ppl == -1:
                continue
            ppl_list.append(ppl)
        else:
            raise TypeError("Wrong text type")
    f.close()

    avg_ppl = sum(ppl_list) / len(ppl_list)
    print("avg_ppl:{}".format(float('{:.2f}'.format(avg_ppl))))

    

