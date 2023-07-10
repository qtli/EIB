import torch
import logging
from torch.utils.data import Dataset
import random
import pandas as pd
import pdb
import ast
import numpy as np
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


class DatasetHelper(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 data_path,
                 special_tokens,
                 sample_max_length=300,
                 exp_max_length=300,
                 do_generate=False,
                 use_cuda=True
                 ):
        self.args = args
        self.tokenizer = tokenizer
        self.special_tokens = special_tokens
        self.data_path = data_path
        self.do_generate = do_generate
        self.device = 'cuda' if use_cuda else 'cpu'

        self.sample_max_length = sample_max_length  #  sin+sout
        self.exp_max_length = exp_max_length   # x

        self.bos = self.tokenizer.bos_token_id
        self.pad = self.tokenizer.pad_token_id
        self.eos = self.tokenizer.eos_token_id
        self.exp = self.tokenizer.encode(special_tokens['additional_special_tokens'][0])[0]  # the output of a sample
        self.exp_ans = self.tokenizer.encode(tokenizer.encode(special_tokens['additional_special_tokens'][1]))[0]

        if args.local_rank in [-1,0]:
            logger.info('bos: {}, eos: {}, pad: {}, exp: {}, ans: {}'.format(self.bos, self.eos, self.pad, self.exp, self.exp_ans))


    def load(self, max_size=256, total_batch_size=0, gpu_num=0):
        df = pd.read_csv(self.data_path)
        self.task_ipt = df['task_ipt']
        self.task_opt = df['task_opt']
        self.expl_ipt = df['expl_ipt']  # concise version
        self.expl_opt = df['expl_opt']

        total_data = len(self.task_ipt)
        total_data = total_data - (total_data % (total_batch_size * gpu_num))
        self.task_ipt = self.task_ipt[:total_data].tolist()
        self.task_opt = self.task_opt[:total_data].tolist()
        self.expl_ipt = self.expl_ipt[:total_data].tolist()
        self.expl_opt = self.expl_opt[:total_data].tolist()

        if max_size is not None:
            max_size = max_size - (max_size % (total_batch_size * gpu_num))
            self.task_ipt = self.task_ipt[:max_size]
            self.task_opt = self.task_opt[:max_size]
            self.expl_ipt = self.expl_ipt[:max_size]
            self.expl_opt = self.expl_opt[:max_size]
    def __len__(self):
        return len(self.task_ipt)

    def __getitem__(self, idx):
        task_ipt = self.task_ipt[idx]
        task_opt = self.task_opt[idx]
        expl_ipt = self.expl_ipt[idx]
        expl_opt = self.expl_opt[idx]


        ##### batch task sample #####
        sin_input = self.special_tokens['bos_token'] + task_ipt + self.special_tokens['bos_token'] + task_opt + self.special_tokens['eos_token']  # <bos> task ipt <bos> task opt <eos>
        # the input format is the same for training and inference, because we replace the generation task with v-information difference

        if not self.args.do_eval:  # training, padding to max
            sin_encodings_dict = self.tokenizer(sin_input, truncation=True, max_length=self.sample_max_length, padding="max_length")
        else:  # test
            sin_encodings_dict = self.tokenizer(sin_input)
        sin_ids = torch.LongTensor(sin_encodings_dict['input_ids'])
        sin_attention_mask = torch.LongTensor(sin_encodings_dict['attention_mask'])
        sin_labels = sin_ids.clone()
        sin_labels.masked_fill_(mask=~sin_attention_mask.bool(), value=-100)


        ##### batch explanation #####
        # 1. for get compressed t vectors
        re_input = self.special_tokens['bos_token'] + expl_ipt + self.special_tokens['bos_token']  # <bos> expl ipt <bos>
        if not self.args.do_eval:  # training, padding to max
            re_encodings_dict = self.tokenizer(re_input, truncation=True, max_length=int((self.exp_max_length)/2), padding="max_length")
        else:
            re_encodings_dict = self.tokenizer(re_input)
        re_ids = torch.LongTensor(re_encodings_dict['input_ids'])
        re_attention_mask = torch.LongTensor(re_encodings_dict['attention_mask'])


        # for predict expl opt
        new_task_ipt = task_ipt.split('<|exp|>')[1].strip()
        expl_task = '<|exp|>' + new_task_ipt + ' ' + task_opt + '<|exp|>'
        new_re_input = re_input.replace('<|exp|>', expl_task)
        if not self.args.do_eval:  ###  training
            new_re_input = new_re_input + expl_opt + self.special_tokens['eos_token']  # <bos> expl ipt <bos> expl opt <eos>
        if not self.args.do_eval:  # training, padding to max
            re_encodings_dict = self.tokenizer(new_re_input, truncation=True, max_length=self.exp_max_length, padding="max_length")
        else:
            re_encodings_dict = self.tokenizer(new_re_input)
        ein_ids = torch.LongTensor(re_encodings_dict['input_ids'])
        ein_attention_mask = torch.LongTensor(re_encodings_dict['attention_mask'])
        ein_labels = ein_ids.clone()
        ein_labels = ein_labels.masked_fill_(mask=~ein_attention_mask.bool(), value=-100)


        # for evaluating ppl when inference
        new_expl_ipt = expl_ipt.replace('<|exp|>', expl_task)
        full_ein_input = self.special_tokens['bos_token'] + new_expl_ipt + self.special_tokens['bos_token'] + expl_opt + self.special_tokens['eos_token']
        full_ein_encodings_dict = self.tokenizer(full_ein_input, truncation=True, max_length=self.exp_max_length, padding="max_length")
        full_ein_ids = torch.LongTensor(full_ein_encodings_dict['input_ids'])
        full_ein_attention_mask = torch.LongTensor(full_ein_encodings_dict['attention_mask'])
        full_ein_labels = full_ein_ids.clone()
        full_ein_labels = full_ein_labels.masked_fill_(mask=~full_ein_attention_mask.bool(), value=-100)


        if self.args.do_eval:
            return {
                'sin_ids': sin_ids,
                'sin_attention_mask': sin_attention_mask,
                'sin_labels': sin_labels,
                're_ids': re_ids,  # for compressed t
                're_attention_mask': re_attention_mask,
                'ein_ids': ein_ids,  # for predict expl opt
                'ein_attention_mask': ein_attention_mask,
                'ein_labels': ein_labels,
                'full_ein_ids': full_ein_ids,
                'full_ein_attention_mask': full_ein_attention_mask,
                'full_ein_labels': full_ein_labels
                    }, task_ipt, task_opt, expl_ipt, expl_opt
        else:
            return {
                'sin_ids': sin_ids,
                'sin_attention_mask': sin_attention_mask,
                'sin_labels': sin_labels,
                're_ids': re_ids,
                're_attention_mask': re_attention_mask,
                'ein_ids': ein_ids,  # for predict expl opt
                'ein_attention_mask': ein_attention_mask,
                'ein_labels': ein_labels,
                'full_ein_ids': full_ein_ids,
                'full_ein_attention_mask': full_ein_attention_mask,
                'full_ein_labels': full_ein_labels
            }

    def print_features(self):
        logger.info("-" * 50 + "Features" + "-" * 50)
        sample_id = random.randint(1, len(self.task_ipt)-2)
        exs = [self.__getitem__(i) for i in range(sample_id, min(sample_id + 2, len(self.task_ipt)))]
        for ex in exs:
            if self.args.do_eval:
                ex = ex[0]
            logger.info("Sample ipt&opt: {}".format(self.tokenizer.decode(ex['sin_ids'].tolist())))
            logger.info("Sample ipt&opt attention mask: {}".format(ex['sin_attention_mask'].tolist()))
            logger.info("Sample ipt&opt label: {}".format(self.tokenizer.decode(ex['sin_labels'].masked_select(ex['sin_labels'] >= 0).tolist())))

            logger.info("Explanation input (to get t): {}".format(self.tokenizer.decode(ex['re_ids'].tolist())))
            logger.info("Explanation input attention mask: {}".format(ex['re_attention_mask'].tolist()))

            logger.info("Explanation ipt&opt: {}".format(self.tokenizer.decode(ex['ein_ids'].tolist())))
            logger.info("Explanation ipt&opt attention mask: {}".format(ex['ein_attention_mask'].tolist()))
            logger.info("Explanation ipt&opt label: {}".format(self.tokenizer.decode(ex['ein_labels'].masked_select(ex['ein_labels'] >= 0).tolist())))

            logger.info("Full explanation ipt&opt: {}".format(self.tokenizer.decode(ex['full_ein_ids'].tolist())))
            logger.info("Full explanation ipt&opt attention mask: {}".format(ex['full_ein_attention_mask'].tolist()))
            logger.info("Full explanation ipt&opt label: {}".format(self.tokenizer.decode(ex['full_ein_labels'].masked_select(ex['full_ein_labels'] >= 0).tolist())))