'''
Dataset we use:
ECQA - CQA with explanations by human annotation, includes positive exp and negative exp.
e-SNLI - Stanford Natural Language Inference dataset with an additional layer of human-annotated natural language explanations of the entailment relations

We use manually designed prompts to steer OPT-13B to generate continuations as explanations
For each sample, we acquire about 3 explanations.
'''
import copy
import pdb
import os
import warnings
warnings.filterwarnings('ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, set_seed
import torch
import torch.nn as nn
import pandas as pd
set_seed(32)
import argparse
import wandb
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('CUDA_VISIBLE_DEVICES')


# ! pip install transformers accelerate
###### DOWNLOAD
from huggingface_hub import snapshot_download
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch, load_checkpoint_in_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np


def download_and_prepare(checkpoint='facebook/opt-30b', to_download=False, do_lower_case=True):
    if to_download:
        # It downloads it to cache and we save the link to be re-used afterwards,
        weights_path = snapshot_download(checkpoint)
        # If the folder contains a checkpoint that isn't sharded, it needs to point to the state dict directly
        # otherwise point to the directory containing the shard
        files = os.listdir(weights_path)
        weights_path = os.path.join(weights_path, 'pytorch_model.bin') if 'pytorch_model.bin' in files else weights_path
    else:
        weights_path = checkpoint

    config = AutoConfig.from_pretrained(weights_path)

    # We then instantiate a configuration, and we load the model from the config inside the init_empty_weights decorator.
    # This decorate instantiates an empty shell with the model. This does not actually load or instantiate any weight, only the shapes.
    # This unties the weights, so we manually retie the weights afterwards.
    # Initializes an empty shell with the model. This is instant and does not take any RAM.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    # Initialize the model under the previous context manager breaks the tied weights.
    model.tie_weights()

    # Finally, we infer an a device map automatically from the model. It will place all the layers to disk, CPU RAM and GPU ram according to the available memory in each device.
    # Infer device map automatically
    device_map = infer_auto_device_map(model.model, no_split_module_classes=["OPTDecoderLayer"], dtype='float16')
    print('device_map: ', device_map)

    if any([k == 'disk' for k in device_map.values()]):
        offload_folder = 'offload_folder'
    else:
        offload_folder = None

    if '30b' in weights_path:
        # Set a few layers to use the disk manually to ensure enough RAM for the 30B checkpoint.
        device_map['decoder.layers.23'] = 'disk'
        device_map['decoder.layers.24'] = 'disk'
        device_map['decoder.layers.25'] = 'disk'
        device_map['decoder.layers.26'] = 'disk'
        device_map['decoder.layers.27'] = 'disk'

    # https://github.com/huggingface/accelerate/issues/362
    load_checkpoint_in_model(
        model.model,
        weights_path,
        device_map=device_map,
        offload_folder=offload_folder,
        dtype='float16',
        offload_state_dict=True
    )
    model.tie_weights()

    full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
    full_model_device_map["lm_head"] = 0
    dispatch_model(model, device_map=full_model_device_map)

    tokenizer = AutoTokenizer.from_pretrained(weights_path, do_lower_case=do_lower_case)

    return model, tokenizer, config

### model parallesm
'''
    ```python
    # Here is an example of a device map on a machine with 4 GPUs using t5-3b, which has a total of 24 attention modules:
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    device_map = {
        0: [0, 1, 2],
        1: [3, 4, 5, 6, 7, 8, 9],
        2: [10, 11, 12, 13, 14, 15, 16],
        3: [17, 18, 19, 20, 21, 22, 23],
    }
    model.parallelize(device_map)
    ```
'''
#### model = AutoModelCausalLM.from_pretrained(weights_path)



def build(weights_path, do_lower_case=True):
    # https://huggingface.co/patrickvonplaten/opt_metaseq_30000m
    model, tokenizer, config = download_and_prepare(checkpoint=weights_path, do_lower_case=do_lower_case)
    return model, tokenizer, config

def infer_by_prompt(
        prompt: str,
        model,
        tokenizer,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        do_sample=False,
        max_length=100,
        num_beams=None):
    # inputs = tokenizer("Hugging Face is pushing the convention that a unicorn with two horns becomes a llama.",
    #                    return_tensors="pt")
    #
    # output = model.generate(inputs["input_ids"].to(0), max_length=50, do_sample=True)

    prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(0)
    if prompt_ids.size(-1) >= max_length:
        max_length += 20
    gen_tokens = model.generate(
        prompt_ids,
        do_sample=do_sample,
        temperature=temperature,
        max_length=max_length,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
    )
    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
    return gen_text



def toy_test(checkpoint='/home/facebook/opt-13b'):
    weights_path = checkpoint
    model, tokenizer, config = build(weights_path=weights_path, do_lower_case=True)  # do_lower_case为False表示不区分大小写
    print('Model built!')
    while True:
        print('input prompt')
        prompt = str(input())
        result = infer_by_prompt(prompt, model, tokenizer, do_sample=False, max_length=50)
        print('no sample: ', result)

        result = infer_by_prompt(prompt, model, tokenizer, do_sample=True, max_length=50)
        print('sample: ', result)

        result = infer_by_prompt(prompt, model, tokenizer, do_sample=True, max_length=50, top_p=0.9)
        print('sample p0.9: ', result)

        pdb.set_trace()


def connection_between_xy(data_path, weights_path, prompt_prefixes:list):
    model, tokenizer, config = build(weights_path, do_lower_case=False)
    print('Model built!')
    df = pd.read_csv(data_path)[:100]
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    ppl_result = {'ppl1': [], 'ppl2': [], 'ppl3': [], 'ppl4': [], 'ppl5': []}
    for index, row in df.iterrows():
        q_text = row['q_text']
        q_ans = row['q_ans']
        exp = row['exp_because']
        exp1, exp2, exp3, exp4, exp5 = exp.split('\t')
        exp_dict = {'ppl1': exp1, 'ppl2': exp2, 'ppl3': exp3, 'ppl4': exp4, 'ppl5': exp5}
        # for prompt in prompt_prefixes:
        for key in exp_dict:
            ipt = q_text + ' ' + exp_dict[key]  # e.g., exp1
            opt = ' So the answer is ' + q_ans
            print(ipt + opt)
            with torch.no_grad():
                ipt_ids = tokenizer(ipt, return_tensors='pt').input_ids.to(0)
                opt_ids = tokenizer(opt, return_tensors='pt').input_ids.to(0)
                new_ipt_ids = torch.cat((ipt_ids, opt_ids), dim=1)
                labels = torch.concat((torch.tensor([[-100] * ipt_ids.size(-1)]).to(0), opt_ids), dim=1)
                results = model(
                    input_ids=new_ipt_ids,
                    labels=labels,
                    return_dict=True)
                loss = results['loss']
                ppl = torch.exp(loss).item()
                if ppl == float('inf'):
                    ppl = 9999999
            ppl_result[key].append(ppl)
            print(ppl)

    for key in ppl_result:
        print(key, np.array(ppl_result[key]).mean())

    delta = []
    for idx in range(100):
        ppls = [ppl_result['ppl1'][idx], ppl_result['ppl2'][idx], ppl_result['ppl3'][idx], ppl_result['ppl4'][idx], ppl_result['ppl5'][idx]]

        minp = 99999999
        maxp = 0
        for p in ppls:
            if p == 9999999:
                continue
            elif p < minp:
                minp = p
            elif p > maxp:
                maxp = p
        if minp - maxp == 99999999:
            continue
        else:
            delta.append(maxp-minp)
    print('delta: ', np.array(delta).mean())



def read_all_esnli(data_path, tofile, weights_path, prompt_prefixes:list, start_idx=0, end_idx=99999):
    '''
    :param path:
    :param prompt_prefixes: ['because','Why?']
    :return:
    '''
    model, tokenizer, config = build(weights_path)
    df = pd.read_csv(data_path)[start_idx:end_idx]
    total_data = len(df)
    exp1_list, exp2_list = [], []
    exp1_why_list, exp2_why_list = [], []

    columns = ['index', 'prompt', 'exp1_' + prompt_prefixes[0], 'exp2_' + prompt_prefixes[0], 'exp1_' + prompt_prefixes[1], 'exp2_' + prompt_prefixes[1]]
    for index, row in df.iterrows():
        q_premise = row['Sentence1']
        q_hypo = row['Sentence2']
        q_label = row['gold_label']
        if q_premise[-1].isalpha() is False:
            q_premise = q_premise[:-1].strip()
        if q_premise.startswith('I ') is False:
            q_premise = q_premise.lower()

        if q_hypo[-1].isalpha() is False:
            q_hypo = q_hypo[:-1].strip()
        if q_hypo.startswith('I ') is False:
            q_hypo = q_hypo.lower()
        print('{} / {}'.format(index, total_data))

        for prefix in prompt_prefixes:
            prompt = 'Let\'s explain a natural language inference. Premise is {}. It is {} to say {}{}'.format(q_premise, q_label.lower(), q_hypo, prefix)
            # prompt = 'If {} It is {} to say {}\n{}'.format(q_premise, q_label, q_hypo, prefix)
            print(prompt)
            # greedy 1 times
            exp1 = infer_by_prompt(prompt=prompt, model=model, tokenizer=tokenizer, do_sample=False, max_length=80)
            if 'because' in prefix:
                exp1_list.append(exp1)  # reserve the very original
            else:
                exp1_why_list.append(exp1)
            print(exp1)

            # sample 4 times
            sample_list = []
            for _ in range(2):
                # exp2 = infer_by_prompt(prompt=prompt, model=model, tokenizer=tokenizer, do_sample=True, max_length=80, top_p=0.9)
                exp2 = infer_by_prompt(prompt=prompt, model=model, tokenizer=tokenizer, do_sample=True, max_length=80)
                sample_list.append(exp2)
            if 'because' in prefix:
                exp2_list.append('\t'.join(sample_list))
            else:
                exp2_why_list.append('\t'.join(sample_list))
            print('\t'.join(sample_list))


        if (index+1) % 50 == 0:
            tmp_data = {
                'exp1_' + prompt_prefixes[0]: exp1_list,
                'exp2_' + prompt_prefixes[0]: exp2_list,
                'exp1_' + prompt_prefixes[1]: exp1_why_list,
                'exp2_' + prompt_prefixes[1]: exp2_why_list,
            }
            exp_df = pd.DataFrame(tmp_data,
                                  columns=['exp1_' + prompt_prefixes[0], 'exp2_' + prompt_prefixes[0], 'exp1_' + prompt_prefixes[1], 'exp2_' + prompt_prefixes[1]])
            exp_df.to_csv(tofile, encoding='utf-8')

    tmp_data = {
        'exp1_'+prompt_prefixes[0]: exp1_list,
        'exp2_'+prompt_prefixes[0]: exp2_list,
        'exp1_' + prompt_prefixes[1]: exp1_why_list,
        'exp2_' + prompt_prefixes[1]: exp2_why_list,
    }
    exp_df = pd.DataFrame(tmp_data, columns=['exp1_'+prompt_prefixes[0], 'exp2_'+prompt_prefixes[0], 'exp1_' + prompt_prefixes[1], 'exp2_' + prompt_prefixes[1]])
    exp_df.to_csv(tofile, encoding='utf-8')



def read_all_ecqa(data_path, tofile, weights_path, prompt_prefixes:list, start_idx=0, end_idx=99999):
    '''
    :param path:
    :param prompt_prefixes: ['because','Why?']
    :return:
    '''
    model, tokenizer, config = build(weights_path)
    df = pd.read_csv(data_path)[start_idx:end_idx]
    total_data = len(df)
    exp1_list, exp2_list = [], []
    exp1_why_list, exp2_why_list = [], []

    for index, row in df.iterrows():
        # q_concept = row['q_concept']  # topic
        q_text = row['q_text']
        q_ans = row['q_ans']
        if q_text.startswith('I ') is False:
            q_text = q_text.lower()

        print('{} / {}'.format(index, total_data))
        for prefix in prompt_prefixes:
            prompt = 'Let\'s explain question and answer. Question is {} Answer is {}{}'.format(q_text.lower(), q_ans.lower(), prefix)
            # prompt = '{} The answer is {} {}'.format(q_text, q_ans, prefix)
            print(prompt)
            # greedy 1 times
            exp1 = infer_by_prompt(prompt=prompt, model=model, tokenizer=tokenizer, do_sample=False, max_length=80)
            # exp1 = exp1.lstrip(prompt)
            # exp1 = max(set(exp1.split('\n\n')),key=len)
            if 'because' in prefix:
                exp1_list.append(exp1)  # reserve the very original
            else:
                exp1_why_list.append(exp1)
            print(exp1)


            # sample 4 times
            sample_list = []
            for _ in range(2):
                exp2 = infer_by_prompt(prompt=prompt, model=model, tokenizer=tokenizer, do_sample=True, max_length=80)
                sample_list.append(exp2)
            if 'because' in prefix:
                exp2_list.append('\t'.join(sample_list))
            else:
                exp2_why_list.append('\t'.join(sample_list))
            print('\t'.join(sample_list))

        if (index+1) % 50 == 0:
            tmp_data = {
                'exp1_' + prompt_prefixes[0]: exp1_list,
                'exp2_' + prompt_prefixes[0]: exp2_list,
                'exp1_' + prompt_prefixes[1]: exp1_why_list,
                'exp2_' + prompt_prefixes[1]: exp2_why_list,
            }
            exp_df = pd.DataFrame(tmp_data, columns=['exp1_' + prompt_prefixes[0], 'exp2_' + prompt_prefixes[0], 'exp1_' + prompt_prefixes[1], 'exp2_' + prompt_prefixes[1]])
            exp_df.to_csv(tofile, encoding='utf-8')

    tmp_data = {
        'exp1_'+prompt_prefixes[0]: exp1_list,
        'exp2_'+prompt_prefixes[0]: exp2_list,
        'exp1_' + prompt_prefixes[1]: exp1_why_list,
        'exp2_' + prompt_prefixes[1]: exp2_why_list,
    }
    exp_df = pd.DataFrame(tmp_data, columns=['exp1_'+prompt_prefixes[0], 'exp2_'+prompt_prefixes[0], 'exp1_' + prompt_prefixes[1], 'exp2_' + prompt_prefixes[1]])
    exp_df.to_csv(tofile, encoding='utf-8')


def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='ecqa', help="task name")
    parser.add_argument("--end_idx", type=int, default=0, help="Id of the current shard")
    parser.add_argument("--start_idx", type=int, default=1, help="Total number of shards")
    parser.add_argument("--checkpoint", type=str, default="YourDICTORY/", help="path to OPT-13B")
    parser.add_argument("--test", type=str, default="ecqa/", help="test set for downstream tasks")
    parser.add_argument("--prompt_result", type=str, default="MixExpl/", help="explanation candidates via prompting")
    parser.add_argument("--new_test", type=str, default="ecqa/", help="new test set for downstream tasks containing samples and explanation candidates")
    args = parser.parse_args()
    return args


def remove_special_tokens(text, special_tokens=['  ', '\n']):
    for s in special_tokens:
        while s in text:
            text = text.replace(s, ' ')
    return text.strip(' ')


def combine_ecqa(
        original_file='/home/qtli/explanation/Explanation_code/data/ECQA/cqa_data_test.csv',
        prompt_file='/home/qtli/explanation/Explanation_code/process_data/get_negative_x/common_unify_data/v3/new_exp_cqa_data_test0913.csv',
        new_file='/home/qtli/explanation/Explanation_code/process_data/get_negative_x/common_unify_data/v3/all_exp_cqa_data_test0913.csv'
):
    # all=pd.read_csv('../../data/ECQA/cqa_data_test.csv')
    all=pd.read_csv(original_file)
    exp=pd.read_csv(prompt_file)

    concept_list = []
    q_text_list = []
    q_ans_list = []
    q_ans_exp_list = []
    q_ans_exp_complex_list = []

    exp_list_because = []
    exp_list_why = []
    c = 0
    for index, row in exp.iterrows():
        allrow = all.iloc[c]
        allrow = dict(allrow)

        q_concept = allrow['q_concept']  # topic
        q_text = allrow['q_text']
        q_ans = allrow['q_ans']
        q_ans_exp = allrow['taskA_pos']  # concise explanation， 'Anxiety is the feeling of fear when doing something with an uncertain outcome.\nApplying for a job has an uncertain outcome.'
        q_ans_exp_complex = allrow['taskB']
        if q_text.startswith('I ') is False:
            q_text = q_text.lower()

        row = dict(row)
        prompt_because = 'Let\'s explain question and answer. Question is {} Answer is {}{}'.format(q_text.lower(), q_ans.lower(), ' because')
        ge = remove_special_tokens(row['exp1_ because'].replace(prompt_because, '').strip(' '))
        newl = [ge]
        se = row['exp2_ because'].split('\t')
        for e in se:
            e = remove_special_tokens(e.replace(prompt_because, '').strip(' '))
            newl.append(e)
        exp_list_because.append('\t'.join(newl))


        prompt_why = 'Let\'s explain question and answer. Question is {} Answer is {}{}'.format(q_text.lower(), q_ans.lower(), '.\nWhy?')
        ge = remove_special_tokens(row['exp1_.\nWhy?'].replace(prompt_why, '').strip(' '))
        newl = [ge]
        se = row['exp2_.\nWhy?'].split('\t')
        for e in se:
            e = remove_special_tokens(e.replace(prompt_why, '').strip(' '))
            newl.append(e)
        exp_list_why.append('\t'.join(newl))

        concept_list.append(q_concept)
        q_text_list.append(q_text)
        q_ans_list.append(q_ans)
        q_ans_exp_list.append(q_ans_exp)
        q_ans_exp_complex_list.append(q_ans_exp_complex)
        c += 1

    tmp_data = {
        'q_concept':concept_list,
        'q_text': q_text_list,
        'q_ans': q_ans_list,
        'taskA_pos': q_ans_exp_list,
        'taskB': q_ans_exp_complex_list,
        'exp_because': exp_list_because,
        'exp_why': exp_list_why
    }
    exp_df = pd.DataFrame(tmp_data, columns=['q_concept', 'q_text', 'q_ans', 'taskA_pos', 'taskB', 'exp_because', 'exp_why'])
    exp_df.to_csv(new_file, encoding='utf-8')

    print(c)



def combine_esnli(
        original_file='',
        prompt_file='',
        new_file=''
):
    gold_label_list = []
    Sentence1_list = []
    Sentence2_list = []
    Explanation_list = []
    exp_list_because = []
    exp_list_why = []

    prompt_df = pd.read_csv(prompt_file)
    df = pd.read_csv(original_file)

    for index, row in df.iterrows():
        q_premise = row['Sentence1']
        q_hypo = row['Sentence2']
        q_label = row['gold_label']
        if q_premise[-1].isalpha() is False:
            q_premise = q_premise[:-1].strip()
        if q_premise.startswith('I ') is False:
            q_premise = q_premise.lower()

        if q_hypo[-1].isalpha() is False:
            q_hypo = q_hypo[:-1].strip()
        if q_hypo.startswith('I ') is False:
            q_hypo = q_hypo.lower()

        p_row = prompt_df.iloc[index]
        prompt_because = 'Let\'s explain a natural language inference. Premise is {}. It is {} to say {}{}'.format(q_premise, q_label.lower(), q_hypo, ' because')
        ge = remove_special_tokens(p_row['exp1_ because'].replace(prompt_because, '').strip(' '))
        newl = [ge]
        se = p_row['exp2_ because'].split('\t')
        for e in se:
            e = remove_special_tokens(e.replace(prompt_because, '').strip(' '))
            newl.append(e)
        exp_list_because.append('\t'.join(newl))



        prompt_why = 'Let\'s explain a natural language inference. Premise is {}. It is {} to say {}{}'.format(q_premise, q_label.lower(), q_hypo, '.\nWhy?')
        ge = remove_special_tokens(p_row['exp1_.\nWhy?'].replace(prompt_why, '').strip(' '))
        newl = [ge]
        se = p_row['exp2_.\nWhy?'].split('\t')
        for e in se:
            e = remove_special_tokens(e.replace(prompt_why, '').strip(' '))
            newl.append(e)
        exp_list_why.append('\t'.join(newl))


        gold_label_list.append(row['gold_label'])
        Sentence1_list.append(row['Sentence1'])
        Sentence2_list.append(row['Sentence2'])
        Explanation_list.append(row['Explanation_1'])


    tmp_data = {
        'gold_label': gold_label_list,
        'Sentence1': Sentence1_list,
        'Sentence2': Sentence2_list,
        'Explanation_1': Explanation_list,
        'exp_because': exp_list_because,
        'exp_why': exp_list_why,
    }

    exp_df = pd.DataFrame(tmp_data, columns=['gold_label', 'Sentence1', 'Sentence2', 'Explanation_1', 'exp_because', 'exp_why'])
    exp_df.to_csv(new_file, encoding='utf-8')



def post_process(
        data_path,
        new_file,
        prefix,
):
    data = pd.read_csv(data_path)

    def remove_incomple_sentence_1(x):
        tmp = x['exp_'+prefix[0]]
        tmps = tmp.split('\t')
        new_tmps_1 = []
        for tmp in tmps:
            if '.' in tmp:
                last_stop_idx = tmp.rindex('.')
                first_part = tmp[:last_stop_idx+1].strip(' ')
                second_part = tmp[last_stop_idx+1:].strip(' ')
                if second_part.lower() in first_part.lower() or len(second_part.split(' '))<=2:  # information redundency, only remove the last subsentence without ending
                    tmp = first_part
            if tmp == '':
                tmp = ' '
            if tmp[-1] != '.' and tmp != ' ':
                tmp += '.'
            new_tmps_1.append(tmp.strip(' '))

        return '\t'.join(new_tmps_1)


    def remove_incomple_sentence_2(x):
        tmp = x['exp_' + prefix[1]]
        tmps = tmp.split('\t')
        new_tmps_2 = []
        for tmp in tmps:
            if '.' in tmp:
                last_stop_idx = tmp.rindex('.')
                first_part = tmp[:last_stop_idx + 1].strip(' ')
                second_part = tmp[last_stop_idx + 1:].strip(' ')
                if second_part.lower() in first_part.lower() or len(second_part.split(' ')) <= 2:  # information redundency, only remove the last subsentence without ending
                    tmp = first_part
            if tmp == '':
                tmp = ' '
            if tmp[-1] != '.' and tmp != ' ':
                tmp += '.'
            new_tmps_2.append(tmp.strip(' '))

        return '\t'.join(new_tmps_2)


    def remove_duplication_1(x):
        tmp = x['exp_'+prefix[0]]
        tmps = tmp.split('\t')
        final_exps_1 = []  ## all exps
        new_tmps_1 = []  ## all substences of all exps

        for tmp in tmps:
            tmp = tmp.split('. ')
            new_tmp = []
            for t in tmp:
                if t.lower().strip(' ').strip('.') not in ' '.join(new_tmps_1).lower():
                    new_tmp.append(t)
                    new_tmps_1.append(t)
            new_tmp = '. '.join(new_tmp)
            if new_tmp == '':
                new_tmp = ' '
            if new_tmp[-1] != '.' and new_tmp != ' ':
                new_tmp += '.'
            final_exps_1.append(new_tmp.strip(' '))

        return '\t'.join(final_exps_1)


    def remove_duplication_2(x):
        tmp = x['exp_' + prefix[1]]
        tmps = tmp.split('\t')
        final_exps_2 = []  ## all exps
        new_tmps_2 = []  ## all substences of all exps

        for tmp in tmps:
            tmp = tmp.split('. ')
            new_tmp = []
            for t in tmp:
                if t.lower().strip(' ').strip('.') not in ' '.join(new_tmps_2).lower():
                    new_tmp.append(t)
                    new_tmps_2.append(t)
            new_tmp = '. '.join(new_tmp)
            if new_tmp == '':
                new_tmp = ' '
            if new_tmp[-1] != '.' and new_tmp != ' ':
                new_tmp += '.'
            final_exps_2.append(new_tmp.strip(' '))

        return '\t'.join(final_exps_2)


    def split_into_list_1(x):
        tmp = x['exp_'+prefix[0]]
        tmps_1 = tmp.split('\t')
        return tmps_1

    def split_into_list_2(x):
        tmp = x['exp_'+prefix[1]]
        tmps_2 = tmp.split('\t')
        return tmps_2

    # ### 1. remove not ending for each explantion candidate
    data['exp_' + prefix[0]] = data.apply(remove_incomple_sentence_1, axis=1)
    data['exp_' + prefix[1]] = data.apply(remove_incomple_sentence_2, axis=1)

    # ### 2. remove repeated substrining for the whole explanation X
    data['exp_'+prefix[0]] = data.apply(remove_duplication_1, axis=1)
    data['exp_' + prefix[1]] = data.apply(remove_duplication_2, axis=1)

    ### 3. split the 5 explanations into a list
    data['exp_'+prefix[0]+'_list'] = data.apply(split_into_list_1, axis=1)
    data['exp_' + prefix[1] + '_list'] = data.apply(split_into_list_2, axis=1)

    data.to_csv(new_file)


if __name__ == '__main__':
    args = parse_params()

    # toy_test(args.checkpoint)

    # connection_between_xy(
    #     data_path=args.test,  # xxx/ECQA/exp_cqa_data_test.csv
    #     weights_path=args.checkpoint, # 'xxx/facebook--opt-13b.main.e55d5e758bcc4ad72d7bb5c1683d4dbad79417f3'
    #     prompt_prefixes=[' is the explanation for '])

    if 'esnli' in args.mode:
        read_all_esnli(
            data_path=args.test,
            tofile=args.prompt_result,
            weights_path=args.checkpoint,
            prompt_prefixes=[' because', '.\nWhy?'],
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

        combine_esnli(
            original_file=args.test,
            prompt_file=args.prompt_result,
            new_file=args.new_test
        )

        post_process(
            data_path=args.new_test,
            new_file=args.new_test,
            prefix=['because', 'why']
        )

    if 'ecqa' in args.mode:
        read_all_ecqa(
            data_path=args.test,
            tofile=args.prompt_result,
            weights_path=args.checkpoint,
            prompt_prefixes=[' because', '.\nWhy?'],
            start_idx=args.start_idx,
            end_idx=args.end_idx,
        )

        combine_ecqa(
                original_file=args.test,
                prompt_file=args.newfile,
                new_file=args.new_test
            )

        post_process(
            data_path=args.new_test,
            new_file=args.new_test,
            prefix=['because', 'why']
        )




