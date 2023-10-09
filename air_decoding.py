
import torch
from modeling_gpt2 import GPT2LMHeadModel
from transformers import GPT2Tokenizer
from tqdm import tqdm
import numpy as np
import random
import argparse
import json
import pdb

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def generate_multi(args):


    f = open("../test_data/Air_{}_{}.jsonl".format(args.task_mode, args.lambda_cs), 'w')
    tokenizer = args.tokenizer
    model = args.model

    for type in args.att_type:
        if args.task_mode == "detoxification":
            if type == '1':
                continue
        for prompt in tqdm(args.prompt):
            with torch.no_grad():
                input_text = torch.tensor([tokenizer(tokenizer.eos_token + prompt).input_ids]).long().to(args.device)
                max_length = args.length
                past_key_values = None
                prev = None
                input_text = input_text.expand(args.samples, input_text.shape[-1])
                if args.task_mode != 'detoxification':
                    cur_len = len(tokenizer.encode(prompt))
                else:
                    cur_len = 0
                result = input_text[:, input_text.shape[-1] - cur_len:]

                att_dic = dict()
                prob = dict()
                sigma_ij = dict()
                cond_logits = dict()
                while cur_len < max_length:
                    if past_key_values is None:
                        dic_base = model(input_ids=input_text, return_dict=True, use_cache=True, use_prefix=False)
                        logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                        for item in args.att_type:
                            if args.task_mode == "sentiment":
                                prefix_id = int(item)
                            elif args.task_mode == "topic":
                                prefix_id = int(item) + 2
                            elif args.task_mode == "detoxification":
                                prefix_id = int(item) + 6

                            att_dic[item] = dict()
                            att_dic[item]['dict'] = model(input_ids=input_text, return_dict=True, use_cache=True, use_prefix=True, prefix_id=prefix_id)
                            att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                            att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values
                            att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)

                            prob[item] = torch.ones(args.samples, 1).to(args.device)
                        
                        for item_i in args.att_type:
                            for item_j in args.att_type:
                                if item_j != item_i:
                                    sigma_ij[item_i + item_j] = prob[item_i] / prob[item_j]

                    else:
                        dic_base = model(input_ids=prev, past_key_values=past_key_values, return_dict=True, use_cache=True, use_prefix=False)
                        logits_base, past_key_values = dic_base.logits[:, -1, :], dic_base.past_key_values

                        for item in args.att_type:
                            if args.task_mode == "sentiment":
                                prefix_id = int(item)
                            elif args.task_mode == "topic":
                                prefix_id = int(item) + 2
                            elif args.task_mode == "detoxification":
                                prefix_id = int(item) + 6

                            att_dic[item]['dict'] = model(input_ids=prev, past_key_values=att_dic[item]['past_kv'], return_dict=True, use_cache=True, use_prefix=True, prefix_id=prefix_id)
                            att_dic[item]['logits'] = att_dic[item]['dict'].logits[:, -1, :]
                            att_dic[item]['past_kv'] = att_dic[item]['dict'].past_key_values

                            prob[item] = torch.gather(att_dic[item]['logits_norm'], dim=-1, index=prev)
                            att_dic[item]['logits_norm'] = -1 / torch.log_softmax(att_dic[item]['logits'], dim=-1)
                        
                        for item_i in args.att_type:
                            for item_j in args.att_type:
                                if item_j != item_i:
                                    sigma_ij[item_i + item_j] *= prob[item_i] / prob[item_j]
                    
                    logits_norm_base = torch.softmax(logits_base, dim=-1)

                    for item_i in args.att_type:
                        cond_logits[item_i] = None
                        for item_j in args.att_type:
                            if cond_logits[item_i] == None:
                                if item_j == item_i:
                                    cond_logits[item_i] = att_dic[item_j]['logits_norm']
                                else:
                                    cond_logits[item_i] = att_dic[item_j]['logits_norm'] * sigma_ij[item_j + item_i]
                            else:
                                if item_j == item_i:
                                    cond_logits[item_i] = cond_logits[item_i] + att_dic[item_j]['logits_norm']
                                else:
                                    cond_logits[item_i] = cond_logits[item_i] + att_dic[item_j]['logits_norm'] * sigma_ij[item_j + item_i]

                        cond_logits[item_i] = att_dic[item_i]['logits_norm'] / cond_logits[item_i]
                        cond_logits[item_i] = torch.nan_to_num(cond_logits[item_i], nan=0)
                    
                    cond_att_logits = cond_logits[type]

                    next_token_logits = logits_norm_base * (cond_att_logits ** args.lambda_cs)

                    top_probs, top_indices = torch.topk(next_token_logits, args.topk, dim=-1)

                    try:
                        tmp_prev = torch.multinomial(top_probs, num_samples=1)
                    except:
                        raise Exception("Too high lambda_cs")
                    prev = top_indices.gather(-1, tmp_prev)
                    result = torch.cat((result, prev), dim=-1)

                    cur_len = cur_len + 1
            
            clean_res = []
            for i in range(args.samples):
                clean_res.append(tokenizer.decode(result[i]))

            if args.task_mode != 'detoxification':
                for i, text in enumerate(clean_res):
                    data = {}
                    data['text'] = text
                    data[args.task_mode] = args.task_att[args.task_mode][type]
                    json.dump(data, f)
                    f.write('\n')
            else:
                data = dict()
                data['prompt'] = prompt
                data['text'] = dict()
                for i, text in enumerate(clean_res):
                    data['text'][i] = text
                json.dump(data, f)
                f.write('\n')

    f.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--prompt_sent", default='../dataset/sentiment-imdb/prompt_sent.jsonl', type=str, help="used for sentiment control task")
    parser.add_argument("--prompt_topic", default='../dataset/topic-agnews/prompt_topic.jsonl', type=str, help="used for topic control task")
    parser.add_argument("--prompt_detoxification", default='../dataset/detoxification-jigsaw/prompt_detoxification.jsonl', type=str, help="used for detoxification task")
    parser.add_argument("--length", default=50, type=int)
    parser.add_argument("--samples", default=20, type=int)
    parser.add_argument("--task_mode", default='sentiment', type=str, choices=['sentiment', 'topic', 'detoxification'])
    parser.add_argument("--att_type", default=['0', '1'], help='sentiment:0 in positive and 1 in negative; topic: 0 in world, 1 in sports, 2 in business, and 3 in science; detoxification: 0 in nontoxic and 1 in toxic')
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--topk", default=200, type=int)
    parser.add_argument("--lambda_cs", default=20.0, type=float, help="control strength")
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--device_num", default='0', type=str)
    args = parser.parse_args()
    args.device = 'cpu' if args.no_cuda else torch.device("cuda:{}".format(args.device_num))
    
    set_seed(args)

    args.model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path).to(args.device)
    args.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    if args.task_mode == 'sentiment':
        args.att_type = ['0', '1']
        prompt_list = list()
        f = open(args.prompt_sent, 'r')
        for item in f.readlines():
            dic = json.loads(item)
            prompt = dic['prompt']
            prompt_list.append(prompt)
        args.prompt = prompt_list

    elif args.task_mode == 'topic':
        args.att_type = ['0', '1', '2', '3']
        prompt_list = list()
        f = open(args.prompt_topic, 'r')
        for item in f.readlines():
            dic = json.loads(item)
            prompt = dic['prompt']
            prompt_list.append(prompt)
        args.prompt = prompt_list

    elif args.task_mode == 'detoxification':
        args.att_type = ['0', '1']
        prompt_list = list()
        f = open(args.prompt_detoxification, 'r')
        for item in f.readlines():
            dic = json.loads(item)
            prompt = dic['prompt']
            prompt_list.append(prompt)
        args.prompt = prompt_list
        
    task_att = dict()
    task_att['sentiment'] = dict()
    task_att['sentiment']['0'] = 'Positive'
    task_att['sentiment']['1'] = 'Negative'
    task_att['topic'] = dict()
    task_att['topic']['0'] = 'World'
    task_att['topic']['1'] = 'Sports'
    task_att['topic']['2'] = 'Business'
    task_att['topic']['3'] = 'Science'
    task_att['detoxification'] = dict()
    task_att['detoxification']['0'] = 'nontoxic'
    task_att['detoxification']['1'] = 'toxic'
    args.task_att = task_att

    generate_multi(args)
