import argparse
import json
from transformers import RobertaTokenizer

def count_ngram(hyps_resp, n):
    """
    Count the number of unique n-grams
    :param hyps_resp: list, a list of responses
    :param n: int, n-gram
    :return: the number of unique n-grams in hyps_resp
    """
    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    ngram = set()
    for resp in hyps_resp:
        if len(resp) < n:
            continue
        for i in range(len(resp) - n + 1):
            ngram.add(' '.join(resp[i: i + n]))
    return len(ngram)

def eval_distinct(hyps_resp, tokenizer):
    """
    compute distinct score for the hyps_resp
    :param hyps_resp: list, a list of hyps responses
    :return: average distinct score for 1, 2-gram
    """

    hyps_resp = [list(map(str, tokenizer.encode(h))) for h in hyps_resp]

    if len(hyps_resp) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    if type(hyps_resp[0]) != list:
        print("ERROR, eval_distinct takes in a list of <class 'list'>, get a list of {} instead".format(
            type(hyps_resp[0])))
        return

    hyps_resp = [(' '.join(i)).split() for i in hyps_resp]
    num_tokens = sum([len(i) for i in hyps_resp])
    dist1 = count_ngram(hyps_resp, 1) / float(num_tokens)
    dist2 = count_ngram(hyps_resp, 2) / float(num_tokens)
    dist3 = count_ngram(hyps_resp, 3) / float(num_tokens)

    return dist1, dist2, dist3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default=None, type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    f = open(args.dataset_path, 'r')
    text_list = []
    for item in f.readlines():
        dic = json.loads(item)
        text = dic['text']
        if isinstance(text, dict):
            for i in range(len(text)):
                text_list.append(text[str(i)])
        elif isinstance(text, str):
            text_list.append(text)
        else:
            raise TypeError("Wrong text Type")
    f.close()

    dist1, dist2, dist3 = eval_distinct(hyps_resp=text_list, tokenizer=tokenizer)
    print("dist1: {}\ndist2: {}\ndist3: {}".format(float('{:.3f}'.format(dist1)), float('{:.3f}'.format(dist2)), float('{:.3f}'.format(dist3))))
