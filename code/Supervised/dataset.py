import torch
import logging
from torch.utils.data import Dataset
import random
import pandas as pd
import pdb
import ast
import copy
import string
punc = string.punctuation
logger = logging.getLogger()

def normalize_case(text):
    if len(text) > 1:
        try:
            normalized = text[0].upper() + text[1:].lower()
            if normalized[-1] != '.':
                normalized = normalized + '.'
        except:
            raise RuntimeError("Cannot normalize text {}".format(text))
        return normalized
    return text


SPECIAL_TOKENS = {"bos_token": "<|bos|>",
                  "eos_token": "<|endoftext|>",
                  "unk_token": "<|u|>",
                  "pad_token": "<|pad|>",
                  "additional_special_tokens": ["<|exp|>", "<|ans|>"],
                  }


class DatasetHelper(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 data_path,
                 special_tokens,
                 max_length=100,
                 do_generate=False,
                 use_cuda=True,
                 ):
        self.args = args
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.do_generate = do_generate
        self.device = 'cuda' if use_cuda else 'cpu'
        self.special_tokens = special_tokens
        self.max_length = max_length
        self.bos = self.tokenizer.bos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.eos = self.tokenizer.eos_token_id
        self.exp = self.tokenizer.encode(special_tokens['additional_special_tokens'][0])[0]  # the output of a sample
        self.exp_ans = self.tokenizer.encode(tokenizer.encode(special_tokens['additional_special_tokens'][1]))[0]
        logger.info('bos: {}, eos: {}, pad: {}, exp: {}, ans: {}'.format(self.bos, self.eos, self.pad, self.exp, self.exp_ans))

        ## vocab_size
    def load(self, max_size=256, total_batch_size=0, gpu_num=0, start_idx=0, end_idx=9999999):
        df = pd.read_csv(self.data_path)[start_idx: end_idx]
        self.human_exp = df['taskA_pos']  # concise version
        self.sins = df['q_text']
        self.souts = 'The answer is ' + df['q_ans'] + '.'

        total_data = len(self.sins)
        total_data = total_data - (total_data % (total_batch_size * gpu_num))  # batch_size_per_gpu:2, gpu:8, total_batch_size:16,
        self.sins = self.sins[:total_data].tolist()
        self.souts = self.souts[:total_data].tolist()
        self.human_exp = self.human_exp[:total_data].tolist()

        if max_size is not None:
            max_size = max_size - (max_size % (total_batch_size * gpu_num))  # batch_size_per_gpu:2, gpu:8, total_batch_size:16,
            self.sins = self.sins[:max_size]
            self.souts = self.souts[:max_size]
            self.human_exp = self.human_exp[:max_size]

    def __len__(self):
        return len(self.sins)

    def __getitem__(self, idx):
        sample = self.sins[idx] + ' ' + self.souts[idx]
        human_exp = self.human_exp[idx]

        input = self.special_tokens['bos_token'] + sample
        if not self.args.do_eval:
            input = input + human_exp + self.special_tokens['eos_token']
        if not self.args.do_eval:
            encodings_dict = self.tokenizer(input, truncation=True, max_length=self.max_length, padding="max_length")
        else:
            encodings_dict = self.tokenizer(input)
        input_ids = torch.LongTensor(encodings_dict['input_ids'])
        attention_mask = torch.LongTensor(encodings_dict['attention_mask'])
        lebels = input_ids.clone()
        lebels.masked_fill_(mask=~attention_mask.bool(), value=-100)

        full_input = self.special_tokens['bos_token'] + sample + human_exp + self.special_tokens['eos_token']
        full_encodings_dict = self.tokenizer(full_input, truncation=True, max_length=self.max_length, padding="max_length")
        full_input_ids = torch.LongTensor(full_encodings_dict['input_ids'])
        full_attention_mask = torch.LongTensor(full_encodings_dict['attention_mask'])
        full_labels = full_input_ids.clone()
        full_labels.masked_fill_(mask=~full_attention_mask.bool(), value=-100)
        if self.args.do_eval:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': lebels,
                'full_input_ids': full_input_ids,
                'full_attention_mask': full_attention_mask,
                'full_labels': full_labels
                    }, sample.replace('\n', '<n>'), human_exp.replace('\n', '<n>')
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': lebels,
                'full_input_ids': full_input_ids,
                'full_attention_mask': full_attention_mask,
                'full_labels': full_labels
                    }

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 500)
        exs = [self.__getitem__(i) for i in range(sample_id, min(sample_id + 3, len(self.sins)))]
        for ex in exs:
            if self.args.do_eval:
                ex = ex[0]
            logger.info("input: {}".format(self.tokenizer.decode(ex['input_ids'].tolist())))
            logger.info("attention mask: {}".format(ex['attention_mask'].tolist()))
            logger.info("labels: {}".format(self.tokenizer.decode(ex['labels'].masked_select(ex['labels'] >= 0).tolist())))
            logger.info("full input: {}".format(self.tokenizer.decode(ex['full_input_ids'].tolist())))
            logger.info("full attention mask: {}".format(ex['full_attention_mask'].tolist()))
            logger.info("full labels: {}".format(self.tokenizer.decode(ex['full_labels'].masked_select(ex['full_labels'] >= 0).tolist())))



class DatasetHelper_esnli(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 data_path,
                 special_tokens,
                 max_length=100,
                 do_generate=False,
                 use_cuda=True,
                 ):
        self.args = args
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.do_generate = do_generate
        self.device = 'cuda' if use_cuda else 'cpu'
        self.special_tokens = special_tokens
        self.max_length = max_length
        self.bos = self.tokenizer.bos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.eos = self.tokenizer.eos_token_id
        self.exp = self.tokenizer.encode(special_tokens['additional_special_tokens'][0])[0]  # the output of a sample
        self.exp_ans = self.tokenizer.encode(tokenizer.encode(special_tokens['additional_special_tokens'][1]))[0]
        logger.info('bos: {}, eos: {}, pad: {}, exp: {}, ans: {}'.format(self.bos, self.eos, self.pad, self.exp, self.exp_ans))

    def load(self, max_size=256, total_batch_size=0, gpu_num=0, start_idx=0, end_idx=9999999):
        df = pd.read_csv(self.data_path)[start_idx: end_idx]
        self.human_exp = df['Explanation_1']  # concise version
        self.sins = 'If ' + df['Sentence1']
        self.souts = 'It is ' + df['gold_label'] + ' to say ' + df['Sentence2']

        total_data = len(self.sins)
        total_data = total_data - (total_data % (total_batch_size * gpu_num))  # batch_size_per_gpu:2, gpu:8, total_batch_size:16,
        self.sins = self.sins[:total_data].tolist()
        self.souts = self.souts[:total_data].tolist()
        self.human_exp = self.human_exp[:total_data].tolist()

        if max_size is not None:
            max_size = max_size - (max_size % (total_batch_size * gpu_num))  # batch_size_per_gpu:2, gpu:8, total_batch_size:16,
            self.sins = self.sins[:max_size]
            self.souts = self.souts[:max_size]
            self.human_exp = self.human_exp[:max_size]

    def __len__(self):
        return len(self.sins)

    def __getitem__(self, idx):
        sample = self.sins[idx] + ' ' + self.souts[idx]
        human_exp = self.human_exp[idx]

        input = self.special_tokens['bos_token'] + sample
        if not self.args.do_eval:
            input = input + human_exp + self.special_tokens['eos_token']
        if not self.args.do_eval:
            encodings_dict = self.tokenizer(input, truncation=True, max_length=self.max_length, padding="max_length")
        else:
            encodings_dict = self.tokenizer(input)
        input_ids = torch.LongTensor(encodings_dict['input_ids'])
        attention_mask = torch.LongTensor(encodings_dict['attention_mask'])
        lebels = input_ids.clone()
        lebels.masked_fill_(mask=~attention_mask.bool(), value=-100)

        full_input = self.special_tokens['bos_token'] + sample + human_exp + self.special_tokens['eos_token']
        full_encodings_dict = self.tokenizer(full_input, truncation=True, max_length=self.max_length, padding="max_length")
        full_input_ids = torch.LongTensor(full_encodings_dict['input_ids'])
        full_attention_mask = torch.LongTensor(full_encodings_dict['attention_mask'])
        full_labels = full_input_ids.clone()
        full_labels.masked_fill_(mask=~full_attention_mask.bool(), value=-100)
        if self.args.do_eval:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': lebels,
                'full_input_ids': full_input_ids,
                'full_attention_mask': full_attention_mask,
                'full_labels': full_labels
                    }, sample.replace('\n', '<n>'), human_exp.replace('\n', '<n>')
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': lebels,
                'full_input_ids': full_input_ids,
                'full_attention_mask': full_attention_mask,
                'full_labels': full_labels
                    }

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, 500)
        exs = [self.__getitem__(i) for i in range(sample_id, min(sample_id + 3, len(self.sins)))]
        for ex in exs:
            if self.args.do_eval:
                ex = ex[0]
            logger.info("input: {}".format(self.tokenizer.decode(ex['input_ids'].tolist())))
            logger.info("attention mask: {}".format(ex['attention_mask'].tolist()))
            logger.info("labels: {}".format(self.tokenizer.decode(ex['labels'].masked_select(ex['labels'] >= 0).tolist())))
            logger.info("full input: {}".format(self.tokenizer.decode(ex['full_input_ids'].tolist())))
            logger.info("full attention mask: {}".format(ex['full_attention_mask'].tolist()))
            logger.info("full labels: {}".format(self.tokenizer.decode(ex['full_labels'].masked_select(ex['full_labels'] >= 0).tolist())))
