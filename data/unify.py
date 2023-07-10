import copy
import pdb
import pandas as pd
import json
from collections import defaultdict
import pickle
import sys
import unittest
import math
import re
import random
import numpy as np


# 1. science exam QA
def science_qa():
    tmp_data = defaultdict(dict)
    science_qa = pd.read_csv('explanation_datasets/science_qa/Elementary-NDMC-Train-WithExplanations.csv')  # 432
    science_qa.dropna(inplace=True)  # 363
    question = science_qa['question'].tolist()  # 'A bird has just hatched from an egg. Which of the following stages most likely comes next in the life cycle of the bird? (A) birth (B) death (C) growth (D) reproduction'
    answer = science_qa['AnswerKey'].tolist()
    expl = science_qa['justification'].tolist()
    assert len(question) == len(answer) == len(expl)
    tmp_data['science_qa'] = {
        'no_split': {
            'question': question,
            'answer_key': answer,
            'explanation': expl,
        }
    }
    print('explanation_for_science_qa: {}'.format(len(expl)))
    sqa = tmp_data['science_qa']['no_split']
    train_size = math.floor(len(sqa['question']) * 0.8)
    dev_size = math.floor((len(sqa['question']) - train_size) / 2)
    test_size = len(sqa['question']) - train_size - dev_size

    train_task_ipt, train_task_opt = [], []
    train_question, train_answer_key, train_explanation = [], [], []
    dev_task_ipt, dev_task_opt = [], []
    dev_question, dev_answer_key, dev_explanation = [], [], []
    test_task_ipt, test_task_opt = [], []
    test_question, test_answer_key, test_explanation = [], [], []
    count = 0
    for q, a, e in list(zip(sqa['question'], sqa['answer_key'], sqa['explanation'])):
        ipt = '[SCIQA]' + q
        na = '(' + a + ') '
        if a != 'D':
            opt = 'The answer is ' + na + q.partition(na)[2].partition('(')[0][:].strip(' ') + '.'
        else:
            opt = 'The answer is ' + na + q.partition(na)[2].partition('.')[0][:].strip(' ') + '.'
        if count < train_size:
            train_task_ipt.append(ipt)
            train_task_opt.append(opt)
            train_explanation.append(e)
            train_question.append(q)
            train_answer_key.append(a)
        elif count >= train_size and count < (train_size+dev_size):
            dev_task_ipt.append(ipt)
            dev_task_opt.append(opt)
            dev_explanation.append(e)
            dev_question.append(q)
            dev_answer_key.append(a)
        else:
            test_task_ipt.append(ipt)
            test_task_opt.append(opt)
            test_explanation.append(e)
            test_question.append(q)
            test_answer_key.append(a)


    new_sqa = {
        'train': {
            'question': train_question,
            'answer_key': train_answer_key,
            'explanation': train_explanation,
            'explanation_opt': train_explanation,
            'task_ipt': train_task_ipt,
            'task_opt': train_task_opt
        },
        'dev': {
            'question': dev_question,
            'answer_key': dev_answer_key,
            'explanation': dev_explanation,
            'explanation_opt': dev_explanation,
            'task_ipt': dev_task_ipt,
            'task_opt': dev_task_opt
        },
        'test': {
            'question': test_question,
            'answer_key': test_answer_key,
            'explanation': test_explanation,
            'explanation_opt': test_explanation,
            'task_ipt': test_task_ipt,
            'task_opt': test_task_opt
        }
    }
    assert len(sqa['question']) == len(train_question) + len(dev_question) + len(test_question)
    tmp_data['science_qa'].update(new_sqa)

    print('train: ', len(train_question))
    print('dev: ', len(dev_question))
    print('test: ', len(test_question))
    return tmp_data


# 2. SEN-MAKING, commonsense validation
def senmaking():
    tmp_data = defaultdict(dict)
    s1 = []
    s2 = []
    label = []
    expls = []
    expl_key = []
    with open('explanation_datasets/senmaking/dataset.jsonl', 'r') as f:
        for line in f.readlines():
            item = json.loads(line)
            s1.append(item['sentence0'])
            s2.append(item['sentence1'])
            label.append(item['false'])
            expls.append([item['A'], item['B'], item['C']])
            expl_key.append(item['reason'])
    tmp_data['senmaking'] = {
        'no_split': {
            's1': s1,
            's2': s2,
            'label': label,
            'explanation': expls,
            'explanation_key': expl_key
        }
    }
    sqa = tmp_data['senmaking']['no_split']
    train_size = math.floor(len(sqa['s1']) * 0.8)
    dev_size = math.floor((len(sqa['s1']) - train_size) / 2)
    test_size = len(sqa['s1']) - train_size - dev_size

    train_s1 = sqa['s1'][:train_size]
    train_s2 = sqa['s2'][:train_size]
    train_label = sqa['label'][:train_size]
    train_expls = sqa['explanation'][:train_size]
    train_expl_key = sqa['explanation_key'][:train_size]

    train_task_ipt = []
    train_task_opt = []
    train_task_expl_ipt = []
    train_task_expl_opt = []
    for i, j, k, m, n in list(zip(train_s1, train_s2, train_label, train_expls, train_expl_key)):
        ipt = '[senmaking]' + 'Sentence 1 is ' + i + '. ' + 'Sentence 2 is ' + j + '.'
        if k == 0:
            opt = 'The true sentence is Sentence 2'
        else:
            opt = 'The true sentence is Sentence 1'
        train_task_ipt.append(ipt)
        train_task_opt.append(opt)
        train_task_expl_ipt.append('\t'.join(m))
        if 'A' in n:
            train_task_expl_opt.append((m[0]))
        elif 'B' in n:
            train_task_expl_opt.append((m[1]))
        else:
            train_task_expl_opt.append((m[2]))

    dev_s1 = sqa['s1'][train_size: train_size + dev_size]
    dev_s2 = sqa['s2'][train_size: train_size + dev_size]
    dev_label = sqa['label'][train_size: train_size + dev_size]
    dev_expls = sqa['explanation'][train_size: train_size + dev_size]
    dev_expl_key = sqa['explanation_key'][train_size: train_size + dev_size]

    dev_task_ipt = []
    dev_task_opt = []
    dev_task_expl_ipt = []
    dev_task_expl_opt = []
    for i, j, k, m, n in list(zip(dev_s1, dev_s2, dev_label, dev_expls, dev_expl_key)):
        ipt = '[senmaking]' + 'Sentence 1 is ' + i + '. ' + 'Sentence 2 is ' + j + '.'
        if k == 0:
            opt = 'The true sentence is Sentence 1'
        else:
            opt = 'The true sentence is Sentence 2'
        dev_task_ipt.append(ipt)
        dev_task_opt.append(opt)
        dev_task_expl_ipt.append('\t'.join(m))
        if n == 'A':
            dev_task_expl_opt.append((m[0]))
        elif n == 'B':
            dev_task_expl_opt.append((m[1]))
        else:
            dev_task_expl_opt.append((m[2]))

    test_s1 = sqa['s1'][train_size + dev_size:]
    test_s2 = sqa['s2'][train_size + dev_size:]
    test_label = sqa['label'][train_size + dev_size:]
    test_expls = sqa['explanation'][train_size + dev_size:]
    test_expl_key = sqa['explanation_key'][train_size + dev_size:]

    test_task_ipt = []
    test_task_opt = []
    test_task_expl_ipt = []
    test_task_expl_opt = []
    for i, j, k, m, n in list(zip(test_s1, test_s2, test_label, test_expls, test_expl_key)):
        ipt = '[senmaking]' + 'Sentence 1 is ' + i + '. ' + 'Sentence 2 is ' + j + '.'
        if k == 0:
            opt = 'The true sentence is Sentence 1'
        else:
            opt = 'The true sentence is Sentence 2'
        test_task_ipt.append(ipt)
        test_task_opt.append(opt)
        test_task_expl_ipt.append('\t'.join(m))
        if n == 'A':
            test_task_expl_opt.append((m[0]))
        elif n == 'B':
            test_task_expl_opt.append((m[1]))
        else:
            test_task_expl_opt.append((m[2]))

    tmp_data['senmaking'].update({
        'train': {
            's1': train_s1,
            's2': train_s2,
            'label': train_label,
            'explanation_list': train_expls,
            'explanation_key': train_expl_key,
            'explanation_ipt': train_task_expl_ipt,
            'explanation_opt': train_task_expl_opt,
            'task_ipt': train_task_ipt,
            'task_opt': train_task_opt
        },
        'dev': {
            's1': dev_s1,
            's2': dev_s2,
            'label': dev_label,
            'explanation_list': dev_expls,
            'explanation_key': dev_expl_key,
            'explanation_ipt': dev_task_expl_ipt,
            'explanation_opt': dev_task_expl_opt,
            'task_ipt': dev_task_ipt,
            'task_opt': dev_task_opt
        },
        'test': {
            's1': test_s1,
            's2': test_s2,
            'label': test_label,
            'explanation_list': test_expls,
            'explanation_key': test_expl_key,
            'explanation_ipt': test_task_expl_ipt,
            'explanation_opt': test_task_expl_opt,
            'task_ipt': test_task_ipt,
            'task_opt': test_task_opt
        }
    })
    print('senmaking: {}'.format(len(s1)))
    print('train: ', len(train_s1))
    print('dev: ', len(dev_s1))
    print('test: ', len(test_s1))

    # total: 2021
    return tmp_data


# 3. Fact checking on news
def liar_plus():
    tmp_data = defaultdict(dict)
    train = open('explanation_datasets/liarplus/dataset/jsonl/train2.jsonl', 'r')
    train_claim = []
    train_label = []
    train_expl = []
    train_task_ipt = []
    train_task_opt = []
    for line in train.readlines():
        item = json.loads(line)
        if item['justification'] is None:
            continue
        train_claim.append(item['claim'])
        train_label.append(item['label'])
        train_expl.append(item['justification'].strip('\n'))
        ipt = '[LIARPLUS]' + item['claim']
        opt = 'The label is ' + item['label']
        train_task_ipt.append(ipt)
        train_task_opt.append(opt)

    val = open('explanation_datasets/liarplus/dataset/jsonl/val2.jsonl', 'r')
    val_claim = []
    val_label = []
    val_expl = []
    val_task_ipt = []
    val_task_opt = []
    for line in val.readlines():
        item = json.loads(line)
        if item['justification'] is None:
            continue
        val_claim.append(item['claim'])
        val_label.append(item['label'])
        val_expl.append(item['justification'].strip('\n'))
        ipt = '[LIARPLUS]' + item['claim']
        opt = 'The label is ' + item['label']
        val_task_ipt.append(ipt)
        val_task_opt.append(opt)

    test = open('explanation_datasets/liarplus/dataset/jsonl/test2.jsonl', 'r')
    test_claim = []
    test_label = []
    test_expl = []
    test_task_ipt = []
    test_task_opt = []
    for line in test.readlines():
        item = json.loads(line)
        if item['justification'] is None:
            continue
        test_claim.append(item['claim'])
        test_label.append(item['label'])
        test_expl.append(item['justification'].strip('\n'))
        ipt = '[LIARPLUS]' + item['claim']
        opt = 'The label is ' + item['label']
        test_task_ipt.append(ipt)
        test_task_opt.append(opt)

    tmp_data['liar_plus'] = {
        'train': {
            'claim': train_claim,
            'label': train_label,
            'explanation': train_expl,
            'explanation_opt': train_expl,
            'task_ipt': train_task_ipt,
            'task_opt': train_task_opt
        },
        'dev': {
            'claim': val_claim,
            'label': val_label,
            'explanation': val_expl,
            'explanation_opt': val_expl,
            'task_ipt': val_task_ipt,
            'task_opt': val_task_opt
        },
        'test': {
            'claim': test_claim,
            'label': test_label,
            'explanation': test_expl,
            'explanation_opt': test_expl,
            'task_ipt': test_task_ipt,
            'task_opt': test_task_opt
        }
    }
    print('liar_plus: {}'.format(len(train_expl) + len(val_expl) + len(test_expl)))
    print('train: ', len(train_expl))
    print('dev: ', len(val_expl))
    print('test: ', len(test_expl))
    '''
    train:  10238
    dev:  1284
    test:  1267
    total:  12789
    '''
    return tmp_data


# 4. Fact checking on pubhealth
def pubhealth():
    tmp_data = defaultdict(dict)
    df_train = pd.read_csv('explanation_datasets/pubhealth/train.tsv', sep='\t')
    df_dev = pd.read_csv('explanation_datasets/pubhealth/dev.tsv', sep='\t')
    df_test = pd.read_csv('explanation_datasets/pubhealth/test.tsv', sep='\t')
    df_train.dropna(subset=['claim', 'label', 'explanation'], inplace=True)
    df_dev.dropna(subset=['claim', 'label', 'explanation'], inplace=True)
    df_test.dropna(subset=['claim', 'label', 'explanation'], inplace=True)

    train_claim = df_train['claim'].tolist()
    train_label = df_train['label'].tolist()
    train_expl = df_train['explanation'].tolist()
    train_task_ipt = ('[PUBHEALTH]' + df_train['claim']).tolist()
    train_task_opt = ('The label is ' + df_train['label']).tolist()

    dev_claim = df_dev['claim'].tolist()
    dev_label = df_dev['label'].tolist()
    dev_expl = df_dev['explanation'].tolist()
    dev_task_ipt = ('[PUBHEALTH]' + df_dev['claim']).tolist()
    dev_task_opt = ('The label is ' + df_dev['label']).tolist()

    test_claim = df_test['claim'].tolist()
    test_label = df_test['label'].tolist()
    test_expl = df_test['explanation'].tolist()
    test_task_ipt = ('[PUBHEALTH]' + df_test['claim']).tolist()
    test_task_opt = ('The label is ' + df_test['label']).tolist()

    tmp_data['pubhealth'] = {
        'train': {
            'claim': train_claim,
            'label': train_label,
            'explanation': train_expl,
            'explanation_opt': train_expl,
            'task_ipt': train_task_ipt,
            'task_opt': train_task_opt
        },
        'dev': {
            'claim': dev_claim,
            'label': dev_label,
            'explanation': dev_expl,
            'explanation_opt': dev_expl,
            'task_ipt': dev_task_ipt,
            'task_opt': dev_task_opt
        },
        'test': {
            'claim': test_claim,
            'label': test_label,
            'explanation': test_expl,
            'explanation_opt': test_expl,
            'task_ipt': test_task_ipt,
            'task_opt': test_task_opt
        }
    }
    print('pubhealth: {}'.format(len(train_expl) + len(dev_expl) + len(test_expl)))
    print('train: ', len(train_expl))
    print('dev: ', len(dev_expl))
    print('test: ', len(test_expl))

    '''
    total:  12253
    train:  9805
    dev:  1215
    test:  1233
    '''
    return tmp_data


# 5. E-δ-NLI
def e_delta_nli():
    tmp_data = defaultdict(dict)
    train = open(
        'explanation_datasets/e_delta_nli/data/defeasible-snli/conceptnet_supervision/train_rationalized_conceptnet.jsonl', 'r')
    train_s1 = []
    train_s2 = []
    train_update = []
    train_expl = []
    train_task_ipt = []
    train_task_opt = []
    train_task_expl = []
    for line in train.readlines():
        item = json.loads(line)
        sign = random.random()
        weak = item['Answer_Attenuator_modifier']
        strength = item['Answer_Intensifier_modifier']
        if (weak == [] and strength == []) or (str(weak) == 'nan' and str(strength) == 'nan') or (
                weak is None and strength is None):
            continue
        weak_expl = item[
            'Attenuator_conceptnet_supervision']  # ['Men is the opposite of boys. Men is the opposite of boy', 'Bad is similar to severe. Trash is bad']
        strength_expl = item[
            'Intensifier_conceptnet_supervision']  # ['Men is the opposite of boys. Men is the opposite of boy', 'The last thing you do when you eat is digest. Stand and digest have similar meanings', 'The last thing you do when you eat is stand up. Stand up and stand have similar meanings']

        #### for weak, (one sample split into two samples)
        if str(weak) != 'nan' and weak != [] and weak is not None:
            train_update.append([weak, strength])
            train_expl.append([weak_expl, strength_expl])
            train_s1.append(item['Input_premise'])
            train_s2.append(item['Input_hypothesis'])
            ipt = '[EDELTANLI]' + 'Premise is ' + item['Input_premise'] + ' ' + 'Hypothesis is ' + item[
                'Input_hypothesis'] + ' ' + 'Update is ' + weak
            opt = 'The label is Weak'
            train_task_ipt.append(ipt)
            train_task_opt.append(opt)
            train_task_expl.append('\t'.join(weak_expl))

        #### for strength, (one sample split into two samples)
        if str(strength) != 'nan' and strength != [] and strength is not None:
            train_update.append([weak, strength])
            train_expl.append([weak_expl, strength_expl])
            train_s1.append(item['Input_premise'])
            train_s2.append(item['Input_hypothesis'])
            ipt = '[EDELTANLI]' + 'Premise is ' + item['Input_premise'] + ' ' + 'Hypothesis is ' + item[
                'Input_hypothesis'] + ' ' + 'Update is ' + strength
            opt = 'The label is Strength'
            train_task_ipt.append(ipt)
            train_task_opt.append(opt)
            train_task_expl.append('\t'.join(strength_expl))

    assert len(train_s1) == len(train_s2) == len(train_update) == len(train_expl) == len(train_task_ipt) == len(
        train_task_opt) == len(train_task_expl)

    dev = open('explanation_datasets/e_delta_nli/data/defeasible-snli/comet_supervision/dev_rationalized_comet.jsonl', 'r')
    dev_s1 = []
    dev_s2 = []
    dev_update = []
    dev_expl = []
    dev_task_ipt = []
    dev_task_opt = []
    dev_task_expl = []
    for line in dev.readlines():
        item = json.loads(line)
        sign = random.random()
        #### for weak, (one sample split into two samples)
        weak = item['Answer_Attenuator_modifier']
        strength = item['Answer_Intensifier_modifier']
        if weak == [] and strength == [] or (str(weak) == 'nan' and str(strength) == 'nan') or (
                weak is None and strength is None):
            continue
        weak_expl = item['Attenuator_comet_supervision']
        strength_expl = item['Intensifier_comet_supervision']
        #### for weak, (one sample split into two samples)
        if str(weak) != 'nan' and weak != [] and weak is not None:
            dev_update.append([weak, strength])
            dev_expl.append([weak_expl, strength_expl])
            dev_s1.append(item['Input_premise'])
            dev_s2.append(item['Input_hypothesis'])
            ipt = '[EDELTANLI]' + 'Premise is ' + item['Input_premise'] + ' ' + 'Hypothesis is ' + item[
                'Input_hypothesis'] + ' ' + 'Update is ' + weak
            opt = 'The label is Weak'
            dev_task_ipt.append(ipt)
            dev_task_opt.append(opt)
            dev_task_expl.append('\t'.join(weak_expl))

        #### for strength, (one sample split into two samples)
        if str(strength) != 'nan' and strength != [] and strength is not None:
            dev_update.append([weak, strength])
            dev_expl.append([weak_expl, strength_expl])
            dev_s1.append(item['Input_premise'])
            dev_s2.append(item['Input_hypothesis'])
            ipt = '[EDELTANLI]' + 'Premise is ' + item['Input_premise'] + ' ' + 'Hypothesis is ' + item[
                'Input_hypothesis'] + ' ' + 'Update is ' + strength
            opt = 'The label is Strength'
            dev_task_ipt.append(ipt)
            dev_task_opt.append(opt)
            dev_task_expl.append('\t'.join(strength_expl))

    test = open('explanation_datasets/e_delta_nli/data/defeasible-snli/comet_supervision/test_rationalized_comet.jsonl', 'r')
    test_s1 = []
    test_s2 = []
    test_update = []
    test_expl = []
    test_task_ipt = []
    test_task_opt = []
    test_task_expl = []
    for line in test.readlines():
        item = json.loads(line)
        sign = random.random()
        weak = item['Answer_Attenuator_modifier']
        strength = item['Answer_Intensifier_modifier']
        if weak == [] and strength == [] or (str(weak) == 'nan' and str(strength) == 'nan') or (
                weak is None and strength is None):
            continue
        weak_expl = item['Attenuator_comet_supervision']
        strength_expl = item['Intensifier_comet_supervision']

        #### for weak, (one sample split into two samples)
        if str(weak) != 'nan' and weak != [] and weak is not None:
            test_update.append([weak, strength])
            test_expl.append([weak_expl, strength_expl])
            test_s1.appexnd(item['Input_premise'])
            test_s2.append(item['Input_hypothesis'])
            ipt = '[EDELTANLI]' + 'Premise is ' + item['Input_premise'] + ' ' + 'Hypothesis is ' + item[
                'Input_hypothesis'] + ' ' + 'Update is ' + weak
            opt = 'The label is ' + 'Weak'
            test_task_ipt.append(ipt)
            test_task_opt.append(opt)
            test_task_expl.append('\t'.join(weak_expl))

        #### for strength, (one sample split into two samples)
        if str(strength) != 'nan' and strength != [] and strength is not None:
            test_update.append([weak, strength])
            test_expl.append([weak_expl, strength_expl])
            test_s1.append(item['Input_premise'])
            test_s2.append(item['Input_hypothesis'])
            ipt = '[EDELTANLI]' + 'Premise is ' + item['Input_premise'] + ' ' + 'Hypothesis is ' + item[
                'Input_hypothesis'] + ' ' + 'Update is ' + strength
            opt = 'The label is ' + 'Strength'
            test_task_ipt.append(ipt)
            test_task_opt.append(opt)
            test_task_expl.append('\t'.join(strength_expl))


    texpls = []
    for e in train_task_expl:
        tmp = str(e).split('\t')
        if '' in tmp:
            tmp.remove('')
        if tmp == []:
            texpls.append('')
            continue
        choose = random.choice(tmp)
        if choose[-1].isalpha():
            choose += '.'
        texpls.append(choose)  # 随机挑一个
    train_task_expl = texpls

    dexpls = []
    for e in dev_task_expl:
        tmp = str(e).split('\t')
        if '' in tmp:
            tmp.remove('')
        if tmp == []:
            dexpls.append('')
            continue
        choose = random.choice(tmp)
        if choose[-1].isalpha():
            choose += '.'
        dexpls.append(choose)  # 随机挑一个
    dev_task_expl = dexpls

    tsexpls = []
    for e in test_task_expl:
        tmp = str(e).split('\t')
        if '' in tmp:
            tmp.remove('')
        if tmp == []:
            tsexpls.append('')
            continue
        choose = random.choice(tmp)
        if choose[-1].isalpha():
            choose += '.'
        tsexpls.append(choose)  # 随机挑一个
    test_task_expl = tsexpls



    tmp_data['e_delta_nli'] = {
        'train': {
            's1': train_s1,
            's2': train_s2,
            'update': train_update,
            'explanation_list': train_expl,
            'task_ipt': train_task_ipt,
            'task_opt': train_task_opt,
            'explanation': train_task_expl,
            'explanation_opt': texpls
        },
        'dev': {
            's1': dev_s1,
            's2': dev_s2,
            'update': dev_update,
            'explanation_list': dev_expl,
            'task_ipt': dev_task_ipt,
            'task_opt': dev_task_opt,
            'explanation': dev_task_expl,
            'explanation_opt': dexpls
        },
        'test': {
            's1': test_s1,
            's2': test_s2,
            'update': test_update,
            'explanation_list': test_expl,
            'task_ipt': test_task_ipt,
            'task_opt': test_task_opt,
            'explanation': test_task_expl,
            'explanation_opt': tsexpls
        }
    }
    print('e_delta_nli: {}'.format(len(train_expl) + len(dev_expl) + len(test_expl)))
    print('train: ', len(train_expl))
    print('dev: ', len(dev_expl))
    print('test: ', len(test_expl))
    '''
    train:  88676
    dev:  1785
    test:  1837
    total:  92298
    '''
    return tmp_data



# 5. e-SNLI
def esnli():
    tmp_data = defaultdict(dict)
    df_train_1 = pd.read_csv('explanation_datasets/esnli/dataset/esnli_train_1.csv')  # 259999
    df_train_2 = pd.read_csv('explanation_datasets/esnli/dataset/esnli_train_2.csv')  # 289368
    trains = [df_train_1, df_train_2]
    df_train = pd.concat(trains, ignore_index=True)
    df_dev = pd.read_csv('explanation_datasets/esnli/dataset/esnli_dev.csv')
    df_test = pd.read_csv('explanation_datasets/esnli/dataset/esnli_test.csv')
    train_s1 = df_train['Sentence1'].tolist()
    train_s2 = df_train['Sentence2'].tolist()
    train_label = df_train['gold_label'].tolist()
    train_expl = []
    if 'Explanation_1' in df_train.columns.tolist():
        train_expl.append(df_train['Explanation_1'].tolist())
        df_train['all_expl'] = df_train['Explanation_1'] + '\t'
    if 'Explanation_2' in df_train.columns.tolist():
        train_expl.append(df_train['Explanation_2'].tolist())
        df_train['all_expl'] += df_train['Explanation_2'] + '\t'
    if 'Explanation_3' in df_train.columns.tolist():
        train_expl.append(df_train['Explanation_3'].tolist())
        df_train['all_expl'] += df_train['Explanation_3']

    df_train['ipt'] = '[ESNLI]' + 'Premise is ' + df_train['Sentence1'] + ' ' + 'Hypothesis is ' + df_train['Sentence2']
    df_train['opt'] = 'The label is ' + df_train['gold_label']
    train_task_ipt = df_train['ipt'].tolist()
    train_task_opt = df_train['opt'].tolist()
    train_task_exp = df_train['all_expl'].tolist()

    dev_s1 = df_dev['Sentence1'].tolist()
    dev_s2 = df_dev['Sentence2'].tolist()
    dev_label = df_dev['gold_label'].tolist()
    dev_expl = []
    if 'Explanation_1' in df_dev.columns.tolist():
        dev_expl.append(df_dev['Explanation_1'].tolist())
        df_dev['all_expl'] = df_dev['Explanation_1'] + '\t'
    if 'Explanation_2' in df_dev.columns.tolist():
        dev_expl.append(df_dev['Explanation_2'].tolist())
        df_dev['all_expl'] += df_dev['Explanation_2'] + '\t'
    if 'Explanation_3' in df_dev.columns.tolist():
        dev_expl.append(df_dev['Explanation_3'].tolist())
        df_dev['all_expl'] += df_dev['Explanation_3']

    df_dev['ipt'] = '[ESNLI]' + 'Premise is ' + df_dev['Sentence1'] + ' ' + 'Hypothesis is ' + df_dev['Sentence2']
    df_dev['opt'] = 'The label is ' + df_dev['gold_label']
    dev_task_ipt = df_dev['ipt'].tolist()
    dev_task_opt = df_dev['opt'].tolist()
    dev_task_exp = df_dev['all_expl'].tolist()

    test_s1 = df_test['Sentence1'].tolist()
    test_s2 = df_test['Sentence2'].tolist()
    test_label = df_test['gold_label'].tolist()
    test_expl = []
    if 'Explanation_1' in df_test.columns.tolist():
        test_expl.append(df_test['Explanation_1'].tolist())
        df_test['all_expl'] = df_test['Explanation_1'] + '\t'
    if 'Explanation_2' in df_test.columns.tolist():
        test_expl.append(df_test['Explanation_2'].tolist())
        df_test['all_expl'] += df_test['Explanation_2'] + '\t'
    if 'Explanation_3' in df_test.columns.tolist():
        test_expl.append(df_test['Explanation_3'].tolist())
        df_test['all_expl'] += df_test['Explanation_3']

    df_test['ipt'] = '[ESNLI]' + 'Premise is ' + df_test['Sentence1'] + ' ' + 'Hypothesis is ' + df_test['Sentence2']
    df_test['opt'] = 'The label is ' + df_test['gold_label']
    test_task_ipt = df_test['ipt'].tolist()
    test_task_opt = df_test['opt'].tolist()
    test_task_exp = df_test['all_expl'].tolist()

    texpls = []
    for e in train_task_exp:
        tmp = str(e).split('\t')
        if '' in tmp:
            tmp.remove('')
        choose = random.choice(tmp)
        if choose[-1].isalpha():
            choose += '.'
        texpls.append(choose)  # 随机挑一个
    train_task_exp = texpls

    dexpls = []
    for e in dev_task_exp:
        tmp = str(e).split('\t')
        if '' in tmp:
            tmp.remove('')
        choose = random.choice(tmp)
        if choose[-1].isalpha():
            choose += '.'
        dexpls.append(choose)  # 随机挑一个
    dev_task_exp = dexpls

    tsexpls = []
    for e in test_task_exp:
        tmp = str(e).split('\t')
        if '' in tmp:
            tmp.remove('')
        choose = random.choice(tmp)
        if choose[-1].isalpha():
            choose += '.'
        tsexpls.append(choose)  # 随机挑一个
    test_task_exp = tsexpls



    tmp_data['esnli'] = {
        'train': {
            's1': train_s1,
            's2': train_s2,
            'label': train_label,
            'explanation_list': train_expl,
            'explanation': train_task_exp,
            'explanation_opt': texpls,
            'task_ipt': train_task_ipt,
            'task_opt': train_task_opt
        },
        'dev': {
            's1': dev_s1,
            's2': dev_s2,
            'label': dev_label,
            'explanation_list': dev_expl,
            'explanation': dev_task_exp,
            'explanation_opt': dexpls,
            'task_ipt': dev_task_ipt,
            'task_opt': dev_task_opt
        },
        'test': {
            's1': test_s1,
            's2': test_s2,
            'label': test_label,
            'explanation_list': test_expl,
            'explanation': test_task_exp,
            'explanation_opt': tsexpls,
            'task_ipt': test_task_ipt,
            'task_opt': test_task_opt
        }
    }
    print('esnli: {}'.format(len(train_s1) + len(dev_s1) + len(test_s1)))
    print('train: ', len(train_s1))
    print('dev: ', len(dev_s1))
    print('test: ', len(test_s1))
    '''
    train:  549367
    dev:  9842
    test:  9824
    total:  569033
    '''
    return tmp_data


# 6. ECQA commonsense QA
def ecqa():
    tmp_data = defaultdict(dict)
    df_train = pd.read_csv('explanation_datasets/ecqa/cqa_data_train.csv')
    df_dev = pd.read_csv('explanation_datasets/ecqa/cqa_data_val.csv')
    df_test = pd.read_csv('explanation_datasets/ecqa/cqa_data_test.csv')

    train_question = df_train['q_text'].tolist()
    train_answer = df_train['q_ans'].tolist()
    train_expl = [df_train['taskA_pos'].tolist(), df_train['taskA_neg'].tolist(), df_train['taskB'].tolist()]
    train_task_ipt = ('[ECQA]' + df_train['q_text']).tolist()
    train_task_opt = ('The answer is ' + df_train['q_ans']).tolist()
    df_train['expl_ipt'] = df_train['taskA_pos'] + '\t' + df_train['taskA_neg']
    train_task_expl_ipt = df_train['expl_ipt'].tolist()
    train_task_expl_opt = df_train['taskB'].tolist()

    dev_question = df_dev['q_text'].tolist()
    dev_answer = df_dev['q_ans'].tolist()
    dev_expl = [df_dev['taskA_pos'].tolist(), df_dev['taskA_neg'].tolist(), df_dev['taskB'].tolist()]
    dev_task_ipt = ('[ECQA]' + df_dev['q_text']).tolist()
    dev_task_opt = ('The answer is ' + df_dev['q_ans']).tolist()
    df_dev['expl_ipt'] = df_dev['taskA_pos'] + '\t' + df_dev['taskA_neg']
    dev_task_expl_ipt = df_dev['expl_ipt'].tolist()
    dev_task_expl_opt = df_dev['taskB'].tolist()

    test_question = df_test['q_text'].tolist()
    test_answer = df_test['q_ans'].tolist()
    test_expl = [df_test['taskA_pos'].tolist(), df_test['taskA_neg'].tolist(), df_test['taskB'].tolist()]
    test_task_ipt = ('[ECQA]' + df_test['q_text']).tolist()
    test_task_opt = ('The answer is ' + df_test['q_ans']).tolist()
    df_test['expl_ipt'] = df_test['taskA_pos'] + '\t' + df_test['taskA_neg']
    test_task_expl_ipt = df_test['expl_ipt'].tolist()
    test_task_expl_opt = df_test['taskB'].tolist()

    tmp_data['ecqa'] = {
        'train': {
            'question': train_question,
            'answer': train_answer,
            'explanation_list': train_expl,
            'explanation_ipt': train_task_expl_ipt,
            'explanation_opt': train_task_expl_opt,
            'task_ipt': train_task_ipt,
            'task_opt': train_task_opt
        },
        'dev': {
            'question': dev_question,
            'answer': dev_answer,
            'explanation_list': dev_expl,
            'explanation_ipt': dev_task_expl_ipt,
            'explanation_opt': dev_task_expl_opt,
            'task_ipt': dev_task_ipt,
            'task_opt': dev_task_opt
        },
        'test': {
            'question': test_question,
            'answer': test_answer,
            'explanation_list': test_expl,
            'explanation_ipt': test_task_expl_ipt,
            'explanation_opt': test_task_expl_opt,
            'task_ipt': test_task_ipt,
            'task_opt': test_task_opt
        }
    }
    print('ecqa: {}'.format(len(train_question) + len(dev_question) + len(test_question)))
    print('train: ', len(train_question))
    print('dev: ', len(dev_question))
    print('test: ', len(test_question))
    '''
    train:  7598
    dev:  1090
    test:  2194
    total:  10882
    '''
    return tmp_data



def unify_all_dataset():
    total_data = {}
    total_data.update(science_qa())
    print('\n')

    total_data.update(senmaking())
    print('\n')

    total_data.update(liar_plus())
    print('\n')

    total_data.update(pubhealth())
    print('\n')

    total_data.update(e_delta_nli())
    print('\n')

    total_data.update(esnli())
    print('\n')

    total_data.update(ecqa())
    print('\n')
    json.dump(total_data, open('explanation_datasets/unify_expl_dataset.json', 'w'), indent=4)



if __name__ == '__main__':
    unify_all_dataset()





