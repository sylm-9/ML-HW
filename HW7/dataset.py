import random
import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    def __init__(self, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = 40
        self.max_paragraph_len = 150
        self.doc_stride = 32
        self.max_seq_len = self.max_question_len + self.max_paragraph_len + 3

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]
        if self.split == "train":
            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])
            start_min = max(0, answer_end_token - self.max_paragraph_len + 1)
            start_max = min(answer_start_token, len(tokenized_paragraph) - self.max_paragraph_len)
            start_max = max(start_min, start_max)
            paragraph_start = random.randint(start_min, start_max + 1)
            paragraph_end = paragraph_start + self.max_paragraph_len
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start: paragraph_end] + [102]
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start
            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), answer_start_token, answer_end_token

        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []
            for i in range(0, len(tokenized_paragraph), self.doc_stride):
                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)
            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):
        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len

        return input_ids, token_type_ids, attention_mask
