import torch
import json


def read_data(file):
    with open(file, 'r', encoding="utf-8") as read:
        data = json.load(read)
    return data["questions"], data["paragraphs"]


def evaluate(data, output, tokenizer):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]
    for k in range(num_of_windows):
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)
        if start_index > end_index:
            continue
        prob = start_prob + end_prob
        if prob > max_prob:
            max_prob = prob
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])
    return answer.replace(' ', '')
