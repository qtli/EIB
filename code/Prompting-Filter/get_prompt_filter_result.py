import json
import pdb
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from nli_demo import setup_before_test as setup_before_test_nli
from nli_demo import get_scores_new as get_scores_nli
from csqa_demo import setup_before_test as setup_before_test_csqa
from csqa_demo import get_scores_new as get_scores_csqa
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def infer_nli(prompt_data_file):
    args, model, tokenizer, data_collator = setup_before_test_nli()
    nli_prompt_data = pd.read_csv(prompt_data_file)

    select_index_list = []
    for index, row in tqdm(nli_prompt_data.iterrows(), total=nli_prompt_data.shape[0]):
        premise = row['Sentence1']
        hypothesis = row['Sentence2']
        label = row['gold_label']
        because_list = row['exp_because'].split('\t')
        why_list = row['exp_why'].split('\t')
        all_list = because_list + why_list

        new_all_list = []
        for b in all_list:
            newb = 'premise: {} hypothesis: {} answer: {}. explanation: because {}'.format(premise, hypothesis, label, b)
            new_all_list.append(newb)
        scores = get_scores_nli(
            new_all_list,
            args.model_type,
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            batch_size=args.batch_size,
            device=args.device,
            verbose=False)
        max_idx = np.argmax(scores)
        select_index_list.append(int(max_idx))

    json.dump(select_index_list, open('esnli_prompt_index.json', 'w'))


def infer_ecqa(prompt_data_file):
    args, model, tokenizer, data_collator = setup_before_test_csqa()


    ecqa_prompt_data = pd.read_csv(prompt_data_file)
    select_index_list = []
    for index, row in tqdm(ecqa_prompt_data.iterrows(), total=ecqa_prompt_data.shape[0]):
        question = row['q_text']
        answer = row['q_ans']
        because_list = row['exp_because'].split('\t')
        why_list = row['exp_why'].split('\t')
        all_list = because_list + why_list

        new_all_list = []
        for b in all_list:
            newb = '{} answer: {}. explanation: {}'.format(question, answer, b)
            new_all_list.append(newb)
        scores = get_scores_csqa(
            new_all_list,
            args.model_type,
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            batch_size=args.batch_size,
            device=args.device,
            verbose=False)
        max_idx = np.argmax(scores)
        select_index_list.append(int(max_idx))
    json.dump(select_index_list, open('ecqa_prompt_index.json', 'w'))


def get_infer_samples(
        data_file='xxx.csv',
        to_file='',
        max_size=99999999,
        type='esnli'
):
    if type == 'esnli':
        index = json.load(open('esnli_prompt_index.json'))[:max_size]
    else:
        index = json.load(open('ecqa_prompt_index.json'))[:max_size]

    nli_prompt_data = pd.read_csv(data_file)
    results = []

    for i, idx in enumerate(index):
        row = nli_prompt_data.iloc[i]
        exp_list = row['exp_because'].split('\t') + row['exp_why'].split('\t')
        results.append(exp_list[idx])

    json.dump(results, open(to_file, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_data_file', type=str, default='../../data/explanation_datasets/esnli/esnli_explanation_cands.csv', help="saving multiple prompting results.")
    parser.add_argument('--filter_result', type=str, default='esnli_pf_result.json')
    parser.add_argument('--type', type=str, default='esnli')
    parser.add_argument('--length', type=int, default=60)
    args = parser.parse_args()


    infer_nli(prompt_data_file=args.prompt_data_file)

    get_infer_samples(
        data_file=args.prompt_data_file,
        to_file=args.filter_result,
        type=args.type
    )


    # combine prompt-filter results and prompt data
    df = pd.read_csv(args.prompt_data_file)
    prompt_filter_result = json.load(open(args.filter_result))
    df['prompt_filter'] = prompt_filter_result
    df.to_csv(args.prompt_data_file)


    ########################################################################################################################

    infer_ecqa(prompt_data_file=args.prompt_data_file)

    get_infer_samples(
        data_file=args.prompt_data_file,
        to_file=args.filter_result,
        type=args.type
    )

    df = pd.read_csv(args.prompt_data_file)
    prompt_filter_result = json.load(open(args.filter_result))
    df['prompt_filter'] = prompt_filter_result
    df.to_csv(args.prompt_data_file)
