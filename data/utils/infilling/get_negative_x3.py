import pdb
from tqdm import tqdm
import pandas as pd
import json
from nltk.corpus import stopwords
import nltk
pos_tag = nltk.pos_tag

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer().lemmatize

import random
import numpy as np
import re
import os
import copy

from fill_blank_glm import glm_setup, glm_generate_one_sample

function_word = [".", ",", "!", "?", " "]  # todo: update label words
all_num, antonym_num = 0, 0
failure_num = {'repeat': 0, 'replace': 0, 'shuffle': 0, 'neg': 0, 'noise': 0, 'para_infill': 0, 'sen_infill': 0}


def write_avail_phrases(
        entity_file="NLEXP-datasets/conceptnet_entity.csv",
        negation_file="NLEXP-datasets/negation.txt",
        antonym_file="NLEXP-datasets/conceptnet_antonym.txt",
        to_file="token_resources.json"
):
    '''从conceptnet中获取有用的信息，比如反义词信息、实体词等'''
    sw = set(stopwords.words('english'))

    avail_phrases = set()
    fin = open(entity_file, 'r')
    for i, line in enumerate(fin):
        avail_phrases.add(' '.join(line.strip().split("|||")[:-1]))
    avail_phrases = avail_phrases - sw
    fin.close()

    fin = open(negation_file, 'r')
    negation_word = []
    for i, line in enumerate(fin):
        word = ' '.join(line.strip().split()[1:])
        negation_word.append(word)
        avail_phrases.add(word)
    fin.close()

    for w in function_word:
        avail_phrases.add(w)

    antonym_word = {}  # 反义词
    with open(antonym_file, "r") as fin:
        for line in fin:
            tmp = line.strip().split("|||")
            if len(tmp) == 3:
                h, t = tmp[0], tmp[2].split()
                if h in antonym_word:
                    antonym_word[h] += t
                else:
                    antonym_word[h] = t[:]

    json.dump([list(avail_phrases), negation_word, antonym_word], open(to_file, 'w'))
    # return avail_phrases, negation_word, antonym_word  # avail_phrases: 604475, negation_word: 33


def write_unify_vocab(
        unify_data,
        avail_phrases,
        unify_vocab=None,
        to_file='unify_vocab.json'
):
    '''for each data, read data, build its vocab (introduce conceptnet)，包含词性、词频'''

    for data_name in unify_data:
        print('dataset: ', data_name)
        if data_name not in unify_vocab:
            unify_vocab[data_name] = {}
        if data_name in ['aquarat_math'] or 'explanation_ipt' in unify_data[data_name]['train'] or 'task_expl_ipt' in unify_data[data_name]['train'] or unify_vocab[data_name]!={}:  # winowhy
            continue
        for split in ['train']:  # 'dev', 'test'
            print('split: ', split)
            split_data = unify_data[data_name][split]
            for i, expl in enumerate(tqdm(split_data['explanation'])):
                expl = expl.strip('\n').strip(' ')
                expls = re.split('\\.|\t', expl)  # 根据 . \t 分割句子，\t是因为处理unify-data时，多个valid candidates用\t连接了
                if i <= 3:
                    print('expl: ', expl)
                    print('split: ', expls)
                for e in expls:
                    e = e.strip(' ')
                    if e == '':
                        continue
                    if e[-1].isalpha():
                        e += ' .'
                    tmp = e.split()
                    pos = pos_tag(tmp)
                    '''
                    tmp:  ['[MALE]', 'was', 'in', 'the', 'bahamas', 'on', 'vacation', '.']
                    pos:  [('[MALE]', 'NN'), ('was', 'VBD'), ('in', 'IN'), ('the', 'DT'), ('bahamas', 'NN'), ('on', 'IN'), ('vacation', 'NN'), ('.', '.')]
                    '''
                    for word_pos in pos:
                        if lemma(word_pos[0], 'v' if word_pos[1][0] == 'V' else 'n') not in avail_phrases:
                            continue
                        if word_pos[0] in unify_vocab[data_name]:  # 这个词
                            unify_vocab[data_name][word_pos[0]]["number"] += 1
                            if word_pos[1] in unify_vocab[data_name][word_pos[0]]:  # 这个词的词性
                                unify_vocab[data_name][word_pos[0]][word_pos[1]] += 1
                            else:
                                unify_vocab[data_name][word_pos[0]][word_pos[1]] = 1
                        else:
                            unify_vocab[data_name][word_pos[0]] = {word_pos[1]: 1, "number": 1}
                # if (i+1) % 50000:
                #     vocab_list = sorted(unify_vocab[data_name], key=lambda x: unify_vocab[data_name][x]["number"], reverse=True)
                #     for v in vocab_list:
                #         v_num = unify_vocab[data_name][v]['number']
                #         unify_vocab[data_name][v].pop('number')
                #         pos_dict = dict(sorted(unify_vocab[data_name][v].items(), key=lambda x: x[1], reverse=True))
                #         pos_dict.update({'number': v_num})
                #         unify_vocab[data_name][v] = pos_dict
                #     json.dump(unify_vocab, open('unify_vocab.json', 'w'))

        vocab_list = sorted(unify_vocab[data_name], key=lambda x: unify_vocab[data_name][x]["number"], reverse=True)
        for v in vocab_list:
            v_num = unify_vocab[data_name][v]['number']
            unify_vocab[data_name][v].pop('number')
            pos_dict = dict(sorted(unify_vocab[data_name][v].items(), key=lambda x:x[1], reverse=True))
            pos_dict.update({'number': v_num})
            unify_vocab[data_name][v] = pos_dict
        json.dump(unify_vocab, open(to_file, 'w'))

    return unify_vocab


def get_avail_phrases(file='token_resources.json'):
    '''与write_avail_phrases()对应'''
    avail_phrases, negation_word, antonym_word = json.load(open(file, 'r'))
    return avail_phrases, negation_word, antonym_word


def get_unify_vocab(file='unify_vocab.json'):
    '''write_unify_vocab()对应'''
    unify_vocab = json.load(open(file, 'r'))
    return unify_vocab



def get_pos_vocab(unify_vocab):
    '''与get_unify_vocab()函数对应，获取词性、词频'''
    pos_vocab_entity = {}
    for data_name in unify_vocab:
        pos_vocab_entity[data_name] = {}
        for word in unify_vocab[data_name]:
            unify_vocab[data_name][word].pop('number')
            pos = unify_vocab[data_name][word]
            for p in pos:
                if p in pos_vocab_entity[data_name]:
                    pos_vocab_entity[data_name][p]["word"].append(word)
                    pos_vocab_entity[data_name][p]["freq"].append(float(pos[p]))
                else:
                    pos_vocab_entity[data_name][p] = {"word": [word], "freq": [float(pos[p])]}
    return pos_vocab_entity


def transfer_expl_to_expls(expl):
    '''因为一个explanation要么是由多个valid candidates组成，要么句子往往比较长，比较多，所以先分成多个sub-sentence，挨个处理'''
    expl = expl.strip('\n').strip(' ')
    expls = re.split('\\.|\t', expl)
    new_expl = []
    for e in expls:
        e = e.strip(' ')
        if e == '':
            continue
        if e[-1].isalpha():
            e += ' .'
        new_expl.append(e)
    return new_expl


def repeat_sentence(expl):
    '''repeat one sentence and delete the original sentence'''
    expls = transfer_expl_to_expls(expl)  # transfer a string into a list by split()

    idx = np.random.choice(
        np.arange(len(expls)),  # 待抽样的list
        1 + int(len(expls)/2),  # size, 3，一半以上的句子都是重复的
        replace=False
    ).tolist()

    s = min(idx)  # 被重复的句子index
    tmp_expl = copy.deepcopy(expls)
    for l in idx:  # 选中的index全部换成要重复的句子内容 #todo: 重复的句子的同时，是否要去除掉一些句子
        tmp_expl[l] = copy.deepcopy(expls[s])
    new_expl = ' '.join(tmp_expl)

    return new_expl


def repeat_ngram(expl):
    '''repeat ngram in one sentence'''
    global failure_num
    def repeat_sen_gram(expls):
        flag = True
        split_sen = []
        idx, pointer_st, pointer_ed = 0, 0, 0
        for _ in range(10):
            try:
                idx = np.random.choice(np.arange(len(expls)))   # 选择某个句子，len(st) [0, 1, ... ,n]
                gram_num = np.random.choice(np.arange(5)[1:])  # 选择ngram，ngram=[1,2,3,4]
                split_sen = expls[idx].strip().split()  # 对这个被选中的句子，分词
                pointer_st = np.random.choice(np.arange(len(split_sen)))  # 选择一个开始点
                pointer_ed = pointer_st + gram_num  # ngram
                if pointer_ed > len(split_sen):  # 如果句子长度不够
                    pointer_ed = pointer_st
                    pointer_st = pointer_ed - gram_num   # 那就往前推
                    if pointer_st < 0:  # 还不够，那就下一个
                        continue
                    else:
                        flag = False
                        break
            except:
                continue
        if flag:  # 没有成功
            failure_num['repeat'] += 1
            return copy.deepcopy(expls)

        sen1, sen2, sen3 = " ".join(split_sen[:pointer_st]), " ".join(split_sen[pointer_st:pointer_ed]), " ".join(split_sen[pointer_ed:])
        tmp_st = copy.deepcopy(expls)
        tmp_st[idx] = " ".join([sen1, sen2, sen2, sen3]).strip()  # 选中的部分（sen2）重复1次

        return tmp_st


    expls = transfer_expl_to_expls(expl)

    for i in range(int(len(expls)/2)):  #  运行句子数目/2次
        expls = repeat_sen_gram(expls)  # input: list, output: list

    new_expl = ' '.join(expls)
    return new_expl


def replace_sentence(expl, data):
    '''
    expl: 要被处理的解释文本
    data: 所有数据
    '''
    global failure_num
    expls = transfer_expl_to_expls(expl)
    tmp_expls = copy.deepcopy(expls)  # 原本的解释

    flag = True
    for _ in range(10):
        try:
            idxs = np.random.choice(
                np.arange(len(expls)),
                np.random.choice(np.arange(1, len(expls))),
                replace=False
            )
            replace_st_id = np.random.choice(np.arange(len(data)))  # 要替换的样本replace_st_id
            for idx in idxs:
                tmp_expls[idx] = np.random.choice(transfer_expl_to_expls(data[replace_st_id]))  # 被替换的句子换成，替换的样本replace_st_id中的某个句子
            flag = False
            break
        except:
            continue
    if flag:  # 没有替换成功
        failure_num['replace'] += 1
        return copy.deepcopy(expl)

    new_expls = ' '.join(tmp_expls)
    return new_expls



def replace_word(expl, antonym_word, pos_vocab_entity, avail_phrases):
    '''
    :param expl: 原始解释文本
    :param antonym_word: 反义词典
    :param pos_vocab_entity: 词性词典
    :param avail_phrases: 关键词典
    :return:
    '''
    global all_num, antonym_num
    global failure_num

    def replace_one_word(expls):
        antonym = False  # 默认没有反义词
        flag_in = True  # 默认没有相同词性的词供替换
        flag = True

        tmp_expls = copy.deepcopy(expls)

        for _ in range(100):  # 尝试100
            tmp_expls = copy.deepcopy(expls)
            idx = np.random.choice(np.arange(len(expls)))  # 选择某个句子
            split_sen = tmp_expls[idx].split()   # 对这个句子分词
            pos_split_sen = pos_tag(split_sen)

            avail_w_id = []
            for w_id, w in enumerate(split_sen):
                if w in avail_phrases and w not in function_word:
                    avail_w_id.append(w_id)
            if len(avail_w_id) == 0:
                continue

            word_id = np.random.choice(avail_w_id)
            if pos_split_sen[word_id][1] not in pos_vocab_entity:   # 寻找类似的词性
                continue

            lemma_word = lemma(pos_split_sen[word_id][0], 'v' if pos_split_sen[word_id][1][0] == 'V' else 'n')
            if lemma_word in antonym_word:  #  如果这个关键词有反义词
                antonym = True
                replace_word = np.random.choice(antonym_word[lemma_word])
            else:  #  如果没有反义词，用相同词性的词替换
                antonym = False
                word_freq = pos_vocab_entity[pos_split_sen[word_id][1]]
                replace_word = ""
                for _ in range(10):  # 尝试10次
                    replace_word = np.random.choice(word_freq["word"], p=word_freq["freq"]/np.sum(word_freq["freq"]))  # 词频越大，被选中的概率越大
                    # if len(word_freq["word"]) == 1 or replace_word != pos_split_sen[word_id][0]:
                    if replace_word != pos_split_sen[word_id][0]:
                        flag_in = False  # 替换成功
                        break

                if flag_in:  # 替换失败
                    replace_word = pos_split_sen[word_id][0]  # 这个词本身，即不替换

            split_sen[word_id] = replace_word
            tmp_expls[idx] = " ".join(split_sen)   # 把词替换后的句子放回列表里
            if antonym or flag_in is False:  # 有反义词，或者被同词性的词替换，结束尝试，退出循环
                flag = False
                break
            else:
                continue

        if flag:  # 尝试失败
            failure_num['replace'] += 1
            return copy.deepcopy(expls), False, True
        return tmp_expls, antonym, flag_in

    expls = transfer_expl_to_expls(expl)
    num = 0
    for idx in np.arange(len(expls)):
        for word in expls[idx].split():
            if word in avail_phrases:
                num += 1
    try:
        final_num = np.random.choice(np.arange(1, int(num*0.15+1)))  # 要替换的词数目，大约15%的关键词被替换为反义词
    except:
        final_num = 1

    for _ in range(final_num):
        expls, antonym, pos = replace_one_word(expls)
        all_num += 1
        if antonym:  # 统计使用反义词替换的词数目
            antonym_num += 1

    new_expls = ' '.join(expls)
    return new_expls


def shuffle_sentence(expl, n_sentence=None):
    '''
    expl: 解释文本
    n_sentence: 打乱的句子数目
    '''
    global failure_num

    def exchange(l, ids, target_ids):
        tmp_l = copy.deepcopy(l)
        for o_id, t_id in zip(ids, target_ids):
            tmp_l[o_id] = copy.deepcopy(l[t_id])
        return tmp_l

    expls = transfer_expl_to_expls(expl)
    if n_sentence is None:
        n_sentence = int(np.random.choice(np.arange(1, len(expls)+1)))  # 最坏的可能是全部打乱

    # exchange n sentences
    flag = True
    tmp_expls = []
    for _ in range(10):
        sen_ids = np.random.choice(np.arange(len(expls)), n_sentence, replace=False)
        target_ids = np.random.permutation(sen_ids)
        tmp_expls = exchange(expls, sen_ids, target_ids)
        if expls != tmp_expls:  # 打乱成功
            flag = False
            break

    if flag:  # 未成功
        failure_num['shuffle'] += 1
        return expl

    new_expls = ' '.join(tmp_expls)
    return new_expls


def change_neg_helper(sen, negation_word):
    def pro(s):
        final_sen = " ".join(s)
        return final_sen

    sen = sen.strip().split()
    for i, n in enumerate(sen):
        if n in negation_word:  # 如果有否定词，直接去掉
            del sen[i]
            return pro(sen)

    neg_list = ["not", "n't"]
    for i, n in enumerate(sen):  # 如果包含情态动词
        if n in ["would", "will", "can", "could", "may", "might", "shall", "should", "do", "does", "did", "am", "is", "are", "was", "were", "be", "been"]:
            sen.insert(i+1, np.random.choice(neg_list))
            return pro(sen)

    pos_sen = pos_tag(sen)  # todo: 只对最早出现的单词进行否定
    for i, n in enumerate(pos_sen):
        if n[1] == "VB":
            sen.insert(i, "do " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBD":
            sen[i] = lemma(sen[i], "v")
            sen.insert(i, "did " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBG":
            sen.insert(i, np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBN":
            sen.insert(i, np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBP":
            sen.insert(i, "do " + np.random.choice(neg_list))
            return pro(sen)
        elif n[1] == "VBZ":
            sen[i] = lemma(sen[i], "v")
            sen.insert(i, "does " + np.random.choice(neg_list))
            return pro(sen)
    return None


def change_neg_sentence(expl, negation_word):
    '''增加或者删除negation words'''
    global failure_num
    flag = True

    expls = transfer_expl_to_expls(expl)
    tmp_expls = copy.deepcopy(expls)

    for _ in range(10):
        try:
            tmp_expls = copy.deepcopy(expls)
            idxs = np.random.choice(np.arange(len(expls)), np.random.choice(np.arange(1, len(expls)+1)), replace=False)
            for idx in idxs:  # 遍历选中的句子
                if expls[idx].isalpha() is False:
                    continue
                tmp_expls_neg = change_neg_helper(expls[idx], negation_word)
                if tmp_expls_neg is not None:
                    tmp_expls[idx] = tmp_expls_neg
                    flag = False
            if flag == False:  # 反义成功
                break
        except:
            continue

    if flag:  # 反义失败
        failure_num['neg'] += 1
        return copy.deepcopy(expl)

    new_expl = ' '.join(tmp_expls)
    return new_expl


def write_general_noise(data_list, to_file='prompt_general_noise.json'):
    noise_dict = {}
    noise_list = set()
    for data in data_list:
        for index, row in data.iterrows():
            prompt = row['exp_because']
            ps = re.split('\\.|\t', prompt)
            for p in ps:
                p = p.strip().lower()
                if p == '' or len(p.split()) < 3:
                    continue
                prefix = ' '.join(p.split()[:5])
                if prefix not in noise_dict:
                    noise_dict[prefix] = {'freq': 0, 'sents': [p]}
                else:
                    noise_dict[prefix]['freq'] += 1
                    noise_dict[prefix]['sents'].append(p)
                noise_list.add(p)

    # rank noise_dict
    sort_noise_dict = dict(sorted(noise_dict.items(), key=lambda x: x[1]['freq'], reverse=True))
    sort_noise_list = sorted(noise_list, key=lambda x: len(x.split()))
    all_data = {
        'sort_noise_dict': sort_noise_dict,
        'sort_noise_list': sort_noise_list
    }
    print('sort_noise_dict: ', len(sort_noise_dict))
    print('sort_noise_list: ', len(sort_noise_list))
    # sort_noise_dict: 20446
    # sort_noise_list: 27639
    json.dump(all_data, open(to_file, 'w'))


def get_general_noise(file):
    noise_data = json.load(open(file, 'r'))
    noise_dict, noise_list = noise_data['sort_noise_dict'], noise_data['sort_noise_list']
    return noise_dict, noise_list


def insert_general_noise(expl, noise_dict, noise_list=None):
    '''
    插入的很有可能与给定样本无关，只是一些具有比较高频prefix的对应句子（也可以叫做noise）
    :param expl:
    :param noise_dict: 根据prefix 频率排序，prefix频率越高排在越前面，同prefix的句子，没有顺序
    :param noise_list: 根据句子长短排序，越短排在越前面
    :return:
    '''
    global failure_num

    expls = transfer_expl_to_expls(expl)
    if len(expls) == 0:
        failure_num['noise'] += 1
        return expl

    idx = np.random.choice(
        np.arange(len(expls)+1),  # 待抽样的list  # todo 有点问题，应该len+1，N个句子，有N+1个slots可以填充inserted noise
        1 + int(len(expls)/2),  # size, 3，一半以上的句子都是重复的
        replace=False
    ).tolist()
    idx.sort(reverse=True)
    idx_num = len(idx)
    noise_samples = np.random.choice(list(noise_dict.keys())[:50], idx_num)  # todo，暂时没考虑noise_list, 统计idx_num是因为"目前每个slot，只插入一个noise"
    # short_noise_samples = np.random.choice(noise_list[: int(len(noise_list)/3)], idx_num-1-int(idx_num/2))

    tmp_expl = copy.deepcopy(expls)
    for i, l in enumerate(idx):  # 选中的index位置插入一些高频的噪音，todo idx必须从后往前遍历 (这个很重要！！！ 或者从前往后，每insert一个内容，后面的index都要+1)
        n = np.random.choice(noise_dict[noise_samples[i]]['sents'])
        if n[-1].isalpha():
           n += ' .'
        tmp_expl.insert(l, n)
    new_expl = ' '.join(tmp_expl)
    return new_expl


def paragraph_infilling(expl, similar_data_dict, before_num=None, after_num=None, infill_maxnum=2):
    '''
    插入的句子与给定样本的某个句子比较相似
    use the off-the-shelf retrieved results by contriever. given a randomly-selected sentence, we insert N similar sentences before or after it.
    :param expl:
    :param similar_data_dict: key is sentence, value is a list of its neighbour sentences
    :param before_num: <3, 1,2
    :param after_num: <3, 1,2
    :return:
    '''
    global failure_num

    expls = transfer_expl_to_expls(expl)
    if len(expls) == 0:
        failure_num['para_infill'] += 1
        return expl

    idx = np.random.choice(
        np.arange(len(expls)),  # 待抽样的list  N个句子
        1 + int(len(expls)/3),  # size, 3，一半以上的句子都是重复的
        replace=False
    ).tolist()
    idx.sort(reverse=False)  # idx必须从前往后遍历, 因为可能涉及在多个位置插入多个sentences，所以需要维护一个offset

    tmp_expl = copy.deepcopy(expls)
    offset = 0
    for i, l in enumerate(idx):  # 选中的index位置插入一些高频的噪音
        # 看看插入的邻居数目
        if before_num is None:
            before_num = random.choice(np.arange(infill_maxnum)) + 1
        if after_num is None:
            after_num = random.choice(np.arange(infill_maxnum)) + 1
        if similar_data_dict[expls[l]] == []:
            continue
        if random.random() < 0.5:  # 从tmp_expl[l+offset] 前面插它的邻居
            to_infill_sentences = np.random.choice(
                similar_data_dict[expls[l]],  # 取样本中的句子，仍然使用expls
                before_num,
                # replace=False
            )
            for j, s in enumerate(to_infill_sentences):
                tmp_expl.insert(l+offset, s['sentence'])
                offset += 1
        else:   # 从后面插入它的邻居句子
            to_infill_sentences = np.random.choice(
                similar_data_dict[expls[l]],
                after_num,
                replace=True
            )
            for j, s in enumerate(to_infill_sentences):
                tmp_expl.insert(l+1+offset, s['sentence'])
                offset += 1
    new_expl = ' '.join(tmp_expl)
    return new_expl



def sentence_infilling(expl, glm_model, glm_tokenizer, glm_args, glm_device):
    '''
    对于一个expl样本中，某些选中的expl子句进行扩充
    :param expl:
    :param glm_model:
    :param glm_tokenizer:
    :param glm_args:
    :param glm_device:
    :return:
    '''
    global failure_num

    expls = transfer_expl_to_expls(expl)
    if len(expls) == 0:
        failure_num['sen_infill'] += 1
        return expl

    tmp_expls = copy.deepcopy(expls)  # 原本的解释

    flag = True
    for _ in range(10):
        try:
            idxs = np.random.choice(
                np.arange(len(expls)),
                np.random.choice(np.arange(1, len(expls))),
                replace=False
            )
            for idx in idxs:

                original_expl = expls[idx]
                # randomly insert [MASK] on the original_expl
                mask_flag = False
                mask_count = 0
                if random.random() < 0.3:  # insert a [MASK] at the beginning
                    original_expl = '[MASK] ' + original_expl.lower()
                    mask_flag = True
                    mask_count += 1
                if random.random() > 0.7:  # insert a [MASK] at the end
                    original_expl = original_expl[:-1].strip() + ' [MASK]'  # [:-1].strip() 去掉标点符号
                    mask_flag = True
                    mask_count += 1
                if mask_flag is False or random.random() < 0.5:   # insert a [MASK] in the middle of the expl  #  or random.random() < 0.5
                    words_c = len(original_expl.split())
                    insert_pos = random.randint(0, words_c)
                    if insert_pos == 0:
                        original_expl = original_expl.lower()
                    if insert_pos == words_c:
                        original_expl = original_expl[:-1].strip()
                    tmp = original_expl.split()
                    tmp.insert(insert_pos, '[MASK]')
                    original_expl = ' '.join(tmp)
                    mask_count += 1
                trim_decode_tokens = glm_generate_one_sample(glm_model, glm_tokenizer, glm_args, glm_device, raw_text=original_expl)
                trim_decode_tokens = trim_decode_tokens.replace("[CLS]", "")
                parts = trim_decode_tokens.split('<|endoftext|>')

                infill_part = parts[0].split('[MASK]')
                res = parts[1].split('<|startofpiece|>')
                insert_idx = 1

                for mi, r in enumerate(res):
                    if r == '':
                        continue
                    new_r = r.replace('\n', '')
                    if len(new_r.split())>36:
                        tmp = new_r[:(new_r.index('.')+1)]
                        new_r = tmp
                    infill_part.insert(insert_idx, new_r)
                    insert_idx += 2
                process_trim_decode_tokens = ' '.join(infill_part).strip().replace('  ', ' ')

                # print('before: ', original_expl)
                # print('after: ', process_trim_decode_tokens)
                # 'The final stage of the life cycle is death, when the bird is no longer alive, generally due to disease, or being[MASK] eaten by a predator .<|endoftext|><|startofpiece|> injured by another bird or'
                # 'The final stage of the life cycle is death, when the bird is no longer alive, generally due to disease, or being injured by another bird or eaten by a predator .'
                # print('see process result !')
                # assert process_trim_decode_tokens != expls[idx], 'We need a inserted spans in the expls[idx] to build process_trim_decode_tokens'
                tmp_expls[idx] = process_trim_decode_tokens
            flag = False
            break
        except:
            continue
    if flag:  # 没有替换成功
        failure_num['sen_infill'] += 1
        return copy.deepcopy(expl)

    new_expls = ' '.join(tmp_expls)
    return new_expls




def write_negatvie_data(
        unify_vocab,
        unify_data,
        type_list,
        time_list,
        time_prob_list,
        type_prob_list,
        antonym_word,
        pos_vocab_entity,
        negation_word,
        avail_phrases,
        noise_dict,
        noise_list,
        similar_data_dict,
        to_file='negative_unify_data.json',
        max_size=None,
        glm_model=None,
        glm_tokenizer=None,
        glm_args=None,
        glm_device=None
):
    print('write negative data !')
    global all_num, antonym_num, failure_num

    if os.path.exists(to_file):   # 在原有的文件上继续构建数据
        negative_unify_data = json.load(open(to_file, 'r'))
        print('load negative_unify_data: ', negative_unify_data.keys())
    else:
        negative_unify_data = {}

    for data_name in unify_data:  # 遍历每个数据集
        print('data_name: ', data_name)

        if data_name in ['aquarat_math'] or data_name not in unify_vocab: #  or unify_vocab[data_name] == {}
            continue  # 如果是数学题数据集，那么先跳过

        negative_unify_data[data_name] = {}
        for split in ['train', 'dev', 'test']:  # 遍历所有子集
            negative_unify_data[data_name][split] = []  # 存储这个子集中每个解释文本对应的负样本
            if split == 'train':
                max_num = 1000
            else:
                max_num = 125
            if data_name in ['ecqa', 'senmaking', 'winowhy']:
                split_data = unify_data[data_name][split]['explanation_opt'][:max_num]  # todo: 每个数据集最多处理50000
            else:
                split_data = unify_data[data_name][split]['explanation'][:max_num]  # todo: 每个数据集最多处理50000

            for idx, expl in enumerate(tqdm(split_data)):
                if idx < 239 and data_name == 'liar_plus' and split == 'train':
                    continue
                expl = str(expl)
                if expl.strip() == '' or expl == 'nan':
                    negative_unify_data[data_name][split].append({
                        'explanation': str(expl),
                        'neg_explanation': [],
                        'neg_op': []
                    })
                    continue

                chaotic_list = np.random.choice(
                    type_list,  # list: ["repeat", "replace", "shuffle", "neg", "noise"]  //  ['para_infill', 'sen_infill']
                    np.random.choice(time_list, p=time_prob_list), # size,  time_list: [1,2,3,4,5],  time_prob_list: [0.2,0.2,0.2,0.2,0.2] //  [1,2,3], [0.4,0.4,0.2]
                    # replace=False,  # without replacement
                    p=type_prob_list
                    # p=type_prob_list / np.sum(type_prob_list)  # type_prob_list: [0.2,0.2,0.2,0.2,0.2]
                ).tolist()  # 对每个样本进行np.random.choice(time_list, p=time_prob_list)次 负样本操作
                nelist = []
                neoplist = []

                # print('chaotic_list: ', chaotic_list)
                for c in chaotic_list:
                    new_expl = ''
                    if c == "repeat":   # we need save each negative operation name (c) in neoplist
                        if random.random() < 0.7:
                            new_expl = repeat_sentence(expl)  # 70%的概率重复句子，expl是个string
                            c += '_sentence'
                        else:
                            new_expl = repeat_ngram(expl)  # 30%的概率重复词
                            c += '_ngram'
                    elif c == "replace":
                        if random.random() < 0.6:
                            new_expl = replace_sentence(expl, unify_data[data_name][split]['explanation_opt'])
                            c += '_sentence'
                        else:
                            new_expl = replace_word(expl, antonym_word, pos_vocab_entity[data_name], avail_phrases)
                            c += '_word'
                    elif c == "shuffle":
                        new_expl = shuffle_sentence(expl, n_sentence=None)
                    elif c == "neg":
                        new_expl = change_neg_sentence(expl, negation_word)
                    elif c == "noise":
                        new_expl = insert_general_noise(expl, noise_dict=noise_dict, noise_list=noise_list)
                    elif c == "para_infill":
                        new_expl = paragraph_infilling(expl, similar_data_dict=similar_data_dict[data_name])   # use the off-the-shelf retrieved results by contriever
                    elif c == "sen_infill":
                        new_expl = sentence_infilling(expl, glm_model, glm_tokenizer, glm_args, glm_device)   # use off-the-shelf pretrained model

                    nelist.append(new_expl)  # new_expl.replace(' .', '.')
                    neoplist.append(c)

                negative_unify_data[data_name][split].append({
                    'explanation': str(expl),
                    'neg_explanation': nelist,
                    'neg_op': neoplist
                })
                if (idx + 1) % 100 == 0:
                    json.dump(negative_unify_data, open(to_file, 'w'), indent=4)

        print("Antonym:", antonym_num)
        print("All:", all_num)
        print("failure_num:", failure_num)

        ### save once after finishing a dataset
        json.dump(negative_unify_data, open(to_file, 'w'), indent=4)

    json.dump(negative_unify_data, open(to_file, 'w'), indent=4)



def combine_negative_data_old(
        unify_data,
        negative_data,
        to_file='unify_expl_dataset.json'
):
    '''
    :param unify_data: original unify data
    :param negative_data: includes explanation (string) and neg_explanation (list)
    :param to_file:
    :return:
    '''
    for data_name in negative_data:
        print('data_name: ', data_name)
        for split in negative_data[data_name]:  # ['train', 'dev', 'test']
            print('split: ', split)
            tmp = copy.deepcopy(unify_data[data_name][split])

            if data_name in ['ecqa', 'senmaking', 'winowhy']:
                tmp['original_explanation_ipt'] = tmp['explanation_ipt']
                tmp['explanation'] = tmp['explanation_opt']


            explanation_ipt = []
            explanation_neg_op = []

            for item in negative_data[data_name][split]:
                explanation_ipt.append(item['neg_explanation'])
                if 'neg_op' not in item:
                    explanation_neg_op.append(item['neg_ops'])
                else:
                    explanation_neg_op.append(item['neg_op'])

            tmp['explanation_ipt'] = explanation_ipt
            tmp['explanation_neg_op'] = explanation_neg_op

            if data_name not in ['ecqa', 'senmaking', 'winowhy']:
                tmp['explanation_opt'] = tmp['explanation']

            unify_data[data_name][split] = tmp
            print('explanation_ipt size: ', len(explanation_ipt))
    json.dump(unify_data, open(to_file, 'w'))




def combine_negative_data(
        unify_data,
        negative_data,
        to_file='unify_expl_dataset.json'
):
    '''
    :param unify_data: original unify data
    :param negative_data: includes explanation (string) and neg_explanation (list)
    :param to_file:
    :return:
    '''
    for data_name in negative_data:
        print('data_name: ', data_name)
        for split in negative_data[data_name]:  # ['train', 'dev', 'test']
            print('split: ', split)
            tmp = copy.deepcopy(unify_data[data_name][split])

            if data_name in ['ecqa', 'senmaking', 'winowhy']:
                tmp['original_explanation_ipt'] = tmp['explanation_ipt']
                tmp['explanation'] = tmp['explanation_opt']


            task_ipt = tmp['task_ipt']
            task_opt = tmp['task_opt']
            explanation_opt = tmp['explanation']


            new_task_ipt = []
            new_task_opt = []
            new_explanation_ipt = []
            new_explanation_opt = []
            new_explanation_neg_op = []

            for idx, item in enumerate(negative_data[data_name][split]):
                opt = explanation_opt[idx]
                if '\t' not in opt:
                    new_task_ipt.append(task_ipt[idx])
                    new_task_opt.append(task_opt[idx])
                    neg_ipt = item['neg_explanation'] # .replace('[MASK]', '').replace('[mask]', '')
                    new_explanation_ipt.append(neg_ipt)
                    new_explanation_opt.append(opt)
                    if 'neg_op' not in item:
                        new_explanation_neg_op.append(item['neg_ops'])
                    else:
                        new_explanation_neg_op.append(item['neg_op'])
                else:
                    opts = opt.split('\t')
                    new_opts = []  # 多个expl references
                    for o in opts:
                        if o[-1].isalpha():
                            o += '.'
                        new_opts.append(o)
                    for o in new_opts:
                        new_task_ipt.append(task_ipt[idx])
                        new_task_opt.append(task_opt[idx])
                        neg_ipt = item['neg_explanation'] # .replace('[MASK]', '').replace('[mask]', '')
                        new_explanation_ipt.append(neg_ipt)
                        new_explanation_opt.append(o)
                        if 'neg_op' not in item:
                            new_explanation_neg_op.append(item['neg_ops'])
                        else:
                            new_explanation_neg_op.append(item['neg_op'])


            new_tmp = {}
            new_tmp['task_ipt'] = new_task_ipt
            new_tmp['task_opt'] = new_task_opt
            new_tmp['explanation_ipt'] = new_explanation_ipt
            new_tmp['explanation_opt'] = new_explanation_opt
            new_tmp['explanation_neg_op'] = new_explanation_neg_op
            assert len(new_task_ipt)==len(new_task_opt)==len(new_explanation_ipt)==len(new_explanation_opt)==len(new_explanation_neg_op)

            unify_data[data_name][split] = new_tmp
            print('explanation_ipt size: ', len(new_explanation_ipt))
    json.dump(unify_data, open(to_file, 'w'))



def mixup_unify_data(
        unify_data,
        to_train='unify_expl_data_train.csv',
        to_dev='unify_expl_data_dev.csv',
        to_test='unify_expl_data_test.csv',
        to_remove_test='unify_expl_data_test_remove.csv',
        remove_data_list=None
):
    train_task_name = []
    train_task_ipt_list = []
    train_task_opt_list = []
    train_expl_ipt_list = []
    train_expl_opt_list = []
    train_expl_neg_op_list = []

    dev_task_name = []
    dev_task_ipt_list = []
    dev_task_opt_list = []
    dev_expl_ipt_list = []
    dev_expl_opt_list = []
    dev_expl_neg_op_list = []

    test_task_name = []
    test_task_ipt_list = []
    test_task_opt_list = []
    test_expl_ipt_list = []
    test_expl_opt_list = []
    test_expl_neg_op_list = []

    remove_test_task_name = []
    remove_test_task_ipt_list = []
    remove_test_task_opt_list = []
    remove_test_expl_ipt_list = []
    remove_test_expl_opt_list = []

    name_map_old = {
        'science_qa': '[SCIQA]',
        'aquarat_math': '[AQUAMATH]',
        'liar_plus': '[LIARPLUS]',
        'esnli': '[ESNLI]',
        'ecqa': '[ECQA]',
        'senmaking': '[senmaking]',
        'pubhealth': '[PUBHEALTH]',
        'winowhy': '[WINOWHY]',
        'e_delta_nli': '[EDELTANLI]'
    }
    name_map_v1 = {
        'science_qa': 'Question answering.',
        'aquarat_math': 'Math problem. ',
        'liar_plus': 'Fact checking. ',
        'esnli': 'Natural language inference. ',
        'ecqa': 'Question answering. ',
        'senmaking': 'Commonsense validataion. ',
        'pubhealth': 'Fact checking. ',
        'winowhy': 'Pronoun coreference resolution. ',
        'e_delta_nli': 'Natural language inference. '
    }
    name_map_v2 = {
        'science_qa': 'This is a science exam question and answer.<|exp|>',
        'aquarat_math': 'This is an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'This is a journalistic claim and veracity label.<|exp|>',
        'esnli': 'This is a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'This is a commonsense question and answer.<|exp|>',
        'senmaking': 'There are two statements and select which one is true.<|exp|>',  #  makes sense
        'pubhealth': 'This is a public health claim and veracity label.<|exp|>',
        'winowhy': 'This is a statement and pronoun coreference resolution result.<|exp|>',
        'e_delta_nli': 'This is a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }

    name_map_v3 = {
        'science_qa': 'Let\'s explain a science exam question and answer.<|exp|>',
        # 'aquarat_math': 'Let\'s explain an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'Let\'s explain a journalistic claim and veracity label.<|exp|>',
        'esnli': 'Let\'s explain a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'Let\'s explain is a commonsense question and answer.<|exp|>',
        'senmaking': 'Let\'s explain two statements where only one statement is true.<|exp|>',  #  makes sense
        'pubhealth': 'Let\'s explain a public health claim and veracity label.<|exp|>',
        'winowhy': 'Let\'s explain a statement and pronoun coreference resolution result.<|exp|>',
        'e_delta_nli': 'Let\'s explain a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }

    for data_name in unify_data:
        train_data = unify_data[data_name]['train']
        dev_data = unify_data[data_name]['dev']
        test_data = unify_data[data_name]['test']
        # special_token_x = name_map_v2[data_name]   # 拼在 解释句子前面
        special_token_x = name_map_v3[data_name]   # 拼在 解释句子前面
        # special_token_y = name_map_v1[data_name]
        special_token_y = name_map_v2[data_name]
        special_token_old = name_map_old[data_name]

        if 'explanation_ipt' not in train_data:
            continue

        if data_name in remove_data_list:   # 比如 ecqa 或者 esnli 的 test，作为 overall 数据的测试集
            test_data = unify_data[data_name]['test']
            ################################ test ###############################
            count = 0
            for idx, items in enumerate(test_data['explanation_ipt']):
                if isinstance(items, str):  # 本身自带explanation_ipt的数据集，比如ecqa
                    items = items.replace(' .', '.').replace('\n', '. ')
                    if items == test_data['explanation_opt'][idx]:
                        continue
                    tmp = special_token_x + items
                    tmp = tmp.replace('\t', '. ').replace('.. ', '. ').strip()
                    if tmp[-1].isalpha():  # 不是以标点符号结束
                        tmp += '.'
                    if (len(tmp.split()) + len(test_data['explanation_opt'][idx].split())) > 250 or len(
                            test_data['explanation_opt'][idx].split()) < 3 or len(tmp.split()) < 3:
                        continue  # 过滤掉太长的，太短的
                    tmp_opt = test_data['explanation_opt'][idx].replace('\t', '. ').replace('.. ', '. ').strip()
                    if tmp_opt[-1].isalpha():
                        tmp_opt += '.'

                    remove_test_task_name.append(data_name)
                    remove_test_expl_ipt_list.append(tmp)
                    tempt = test_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                    remove_test_task_ipt_list.append(tempt)
                    remove_test_task_opt_list.append(test_data['task_opt'][idx])
                    remove_test_expl_opt_list.append(tmp_opt)
                    count += 1
                    # if 100 < idx < 103:
                    #     print('data_name - test: ', data_name)
                    #     print('expl_ipt: ', tmp)
                    #     print('expl_opt: ', tmp_opt)
                else:
                    for jdx, item in enumerate(items):  # 需要自己构造explanation_ipt的数据集
                        item = item.replace(' .', '.')
                        if item == test_data['explanation_opt'][idx]:
                            continue
                        tmp = special_token_x + item
                        if (len(tmp.split()) + len(test_data['explanation_opt'][idx].split())) > 250 or len(
                                test_data['explanation_opt'][idx].split()) < 3 or len(tmp.split()) < 3:
                            continue  # 过滤掉太长的，太短的

                        remove_test_task_name.append(data_name)
                        remove_test_expl_ipt_list.append(tmp)
                        tempt = test_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                        remove_test_task_ipt_list.append(tempt)
                        remove_test_task_opt_list.append(test_data['task_opt'][idx])
                        remove_test_expl_opt_list.append(test_data['explanation_opt'][idx])
                        count += 1
                        # if 100 < idx < 103:
                        #     print('data_name - test: ', data_name)
                        #     print('expl_ipt: ', tmp)
                        #     print('expl_opt: ', test_data['explanation_opt'][idx])
                        # if jdx > 2:
                        #     break
            print('remove data name: ', data_name)
            print('test: ', count)

            print('test size: ', len(remove_test_expl_ipt_list))
            remove_unify_test = list(zip(remove_test_task_ipt_list, remove_test_task_opt_list, remove_test_expl_ipt_list, remove_test_expl_opt_list))
            random.shuffle(remove_unify_test)
            remove_test_task_ipt_list, remove_test_task_opt_list, remove_test_expl_ipt_list, remove_test_expl_opt_list = zip(*remove_unify_test)

            remove_unify_test_data = {
                'task_ipt': list(remove_test_task_ipt_list),
                'task_opt': list(remove_test_task_opt_list),
                'expl_ipt': list(remove_test_expl_ipt_list),
                'expl_opt': list(remove_test_expl_opt_list)
            }

            test_df = pd.DataFrame(remove_unify_test_data, columns=['task_ipt', 'task_opt', 'expl_ipt', 'expl_opt'])
            test_df.to_csv(to_remove_test, encoding='utf-8')
            continue


        print(data_name)
        ################################ train ###############################
        count = 0
        for idx, items in enumerate(train_data['explanation_ipt']):
            if isinstance(items, str):  # 本身自带explanation_ipt的数据集，比如ecqa
                items = items.replace(' .', '.').replace('\n', '. ')
                if items == train_data['explanation_opt'][idx]:
                    continue
                tmp = special_token_x + items
                tmp = tmp.replace('\t', '. ').replace('.. ', '. ').strip()
                if tmp[-1].isalpha():  # 不是以标点符号结束
                    tmp += '.'
                if (len(tmp.split()) + len(train_data['explanation_opt'][idx].split())) > 250 or len(train_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                    continue  # 过滤掉太长的，太短的
                tmp_opt = train_data['explanation_opt'][idx].replace('\t', '. ').replace('.. ', '. ').strip()
                if tmp_opt[-1].isalpha():
                    tmp_opt += '.'

                train_task_name.append(data_name)
                train_expl_ipt_list.append(tmp)
                train_expl_opt_list.append(tmp_opt)
                tempt = train_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                train_task_ipt_list.append(tempt)
                train_task_opt_list.append(train_data['task_opt'][idx])
                count += 1
                # if 100 < idx < 103:
                #     print('data_name -train: ', data_name)
                #     print('expl_ipt: ', tmp)
                #     print('expl_opt: ', tmp_opt)
            else:
                for jdx, item in enumerate(items):  # 需要自己构造explanation_ipt的数据集
                    item = item.replace(' .', '.').replace('\n', '. ').replace('[MASK]', '').replace('[mask]', '')
                    if item == train_data['explanation_opt'][idx]:
                        continue
                    tmp = special_token_x + item
                    if (len(tmp.split()) + len(train_data['explanation_opt'][idx].split())) > 250 or len(train_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                        continue  # 过滤掉太长的，太短的

                    train_task_name.append(data_name)
                    train_expl_ipt_list.append(tmp)
                    train_expl_opt_list.append(train_data['explanation_opt'][idx])
                    tempt = train_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                    train_task_ipt_list.append(tempt)
                    train_task_opt_list.append(train_data['task_opt'][idx])
                    train_expl_neg_op_list.append(train_data['explanation_neg_op'][idx])
                    count += 1
                    # if 100 < idx < 103:
                    #     print('data_name -train : ', data_name)
                    #     print('expl_ipt: ', tmp)
                    #     print('expl_opt: ', train_data['explanation_opt'][idx])
        print('train: ', count)


        ################################ dev ###############################
        count = 0
        for idx, items in enumerate(dev_data['explanation_ipt']):
            if isinstance(items, str):  # 本身自带explanation_ipt的数据集，比如ecqa
                items = items.replace(' .', '.').replace('\n', '. ')
                if items == dev_data['explanation_opt'][idx]:
                    continue
                tmp = special_token_x + items
                tmp = tmp.replace('\t', '. ').replace('.. ', '. ').strip()
                if tmp[-1].isalpha():  # 不是以标点符号结束
                    tmp += '.'
                if (len(tmp.split()) + len(dev_data['explanation_opt'][idx].split())) > 250 or len(dev_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                    continue  # 过滤掉太长的，太短的
                tmp_opt = dev_data['explanation_opt'][idx].replace('\t', '. ').replace('.. ', '. ').strip()
                if tmp_opt[-1].isalpha():
                    tmp_opt += '.'

                dev_task_name.append(data_name)
                dev_expl_ipt_list.append(tmp)
                tempt = dev_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                dev_task_ipt_list.append(tempt)
                dev_task_opt_list.append(dev_data['task_opt'][idx])
                dev_expl_opt_list.append(tmp_opt)
                count += 1
                # if 100 < idx < 103:
                #     print('data_name -dev : ', data_name)
                #     print('expl_ipt: ', tmp)
                #     print('expl_opt: ', tmp_opt)
            else:
                for jdx, item in enumerate(items):  # 需要自己构造explanation_ipt的数据集
                    item = item.replace(' .', '.').replace('\n', '. ').replace('[MASK]', '').replace('[mask]', '')
                    if item == dev_data['explanation_opt'][idx]:
                        continue
                    tmp = special_token_x + item
                    if (len(tmp.split()) + len(dev_data['explanation_opt'][idx].split())) > 250 or len(dev_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                        continue  # 过滤掉太长的，太短的

                    dev_task_name.append(data_name)
                    dev_expl_ipt_list.append(tmp)
                    tempt = dev_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                    dev_task_ipt_list.append(tempt)
                    dev_task_opt_list.append(dev_data['task_opt'][idx])
                    dev_expl_opt_list.append(dev_data['explanation_opt'][idx])
                    dev_expl_neg_op_list.append(dev_data['explanation_neg_op'][idx])
                    count += 1
                    # if 100 < idx < 103:
                    #     print('data_name - dev: ', data_name)
                    #     print('expl_ipt: ', tmp)
                    #     print('expl_opt: ', dev_data['explanation_opt'][idx])
                    # if jdx > 2:
                    #     break
        print('dev: ', count)


        ################################ test ###############################
        count = 0
        for idx, items in enumerate(test_data['explanation_ipt']):
            if isinstance(items, str):  # 本身自带explanation_ipt的数据集，比如ecqa
                items = items.replace(' .', '.').replace('\n', '. ')
                if items == test_data['explanation_opt'][idx]:
                    continue
                tmp = special_token_x + items
                tmp = tmp.replace('\t', '. ').replace('.. ', '. ').strip()
                if tmp[-1].isalpha():  # 不是以标点符号结束
                    tmp += '.'
                if (len(tmp.split()) + len(test_data['explanation_opt'][idx].split())) > 250 or len(test_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                    continue  # 过滤掉太长的，太短的
                tmp_opt = test_data['explanation_opt'][idx].replace('\t', '. ').replace('.. ', '. ').strip()
                if tmp_opt[-1].isalpha():
                    tmp_opt += '.'

                test_task_name.append(data_name)
                test_expl_ipt_list.append(tmp)
                tempt = test_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                test_task_ipt_list.append(tempt)
                test_task_opt_list.append(test_data['task_opt'][idx])
                test_expl_opt_list.append(tmp_opt)
                count += 1
                # if 100 < idx < 103:
                #     print('data_name - test: ', data_name)
                #     print('expl_ipt: ', tmp)
                #     print('expl_opt: ', tmp_opt)
            else:
                for jdx, item in enumerate(items):  # 需要自己构造explanation_ipt的数据集
                    item = item.replace(' .', '.').replace('\n', '. ').replace('[MASK]', '').replace('[mask]', '')
                    if item == test_data['explanation_opt'][idx]:
                        continue
                    tmp = special_token_x + item
                    if (len(tmp.split()) + len(test_data['explanation_opt'][idx].split())) > 250 or len(test_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                        continue  # 过滤掉太长的，太短的

                    test_task_name.append(data_name)
                    test_expl_ipt_list.append(tmp)
                    tempt = test_data['task_ipt'][idx].replace(special_token_old, special_token_y)
                    test_task_ipt_list.append(tempt)
                    test_task_opt_list.append(test_data['task_opt'][idx])
                    test_expl_opt_list.append(test_data['explanation_opt'][idx])
                    test_expl_neg_op_list.append(test_data['explanation_neg_op'][idx])
                    count += 1
                    # if 100 < idx < 103:
                    #     print('data_name - test: ', data_name)
                    #     print('expl_ipt: ', tmp)
                    #     print('expl_opt: ', test_data['explanation_opt'][idx])
                    # if jdx > 2:
                    #     break
        print('test: ', count)

    print('dev size: ', len(dev_expl_ipt_list))
    unify_dev = list(zip(dev_task_name, dev_task_ipt_list, dev_task_opt_list, dev_expl_ipt_list, dev_expl_opt_list, dev_expl_neg_op_list))
    random.shuffle(unify_dev)
    dev_task_name, dev_task_ipt_list, dev_task_opt_list, dev_expl_ipt_list, dev_expl_opt_list, dev_expl_neg_op_list = zip(*unify_dev)

    mixup_unify_dev_data = {
        'task_name': list(dev_task_name),
        'task_ipt': list(dev_task_ipt_list),
        'task_opt': list(dev_task_opt_list),
        'expl_ipt': list(dev_expl_ipt_list),
        'expl_opt': list(dev_expl_opt_list),
        'neg_op': list(dev_expl_neg_op_list)
    }

    dev_df = pd.DataFrame(mixup_unify_dev_data, columns=['task_name', 'task_ipt', 'task_opt', 'expl_ipt', 'expl_opt', 'neg_op'])
    dev_df.to_csv(to_dev, encoding='utf-8')

    print('test size: ', len(test_expl_ipt_list))
    unify_test = list(zip(test_task_name, test_task_ipt_list, test_task_opt_list, test_expl_ipt_list, test_expl_opt_list, test_expl_neg_op_list))
    random.shuffle(unify_test)
    test_task_name, test_task_ipt_list, test_task_opt_list, test_expl_ipt_list, test_expl_opt_list, test_expl_neg_op_list = zip(*unify_test)

    mixup_unify_test_data = {
        'task_name': list(test_task_name),
        'task_ipt': list(test_task_ipt_list),
        'task_opt': list(test_task_opt_list),
        'expl_ipt': list(test_expl_ipt_list),
        'expl_opt': list(test_expl_opt_list),
        'neg_op': list(test_expl_neg_op_list)
    }

    test_df = pd.DataFrame(mixup_unify_test_data, columns=['task_name', 'task_ipt', 'task_opt', 'expl_ipt', 'expl_opt', 'neg_op'])
    test_df.to_csv(to_test, encoding='utf-8')



    print('train size: ', len(train_expl_ipt_list))
    unify_train = list(zip(train_task_name, train_task_ipt_list, train_task_opt_list, train_expl_ipt_list, train_expl_opt_list, train_expl_neg_op_list))
    random.shuffle(unify_train)
    train_task_name, train_task_ipt_list, train_task_opt_list, train_expl_ipt_list, train_expl_opt_list, train_expl_neg_op_list = zip(*unify_train)

    mixup_unify_train_data = {
        'task_name': list(train_task_name),
        'task_ipt': list(train_task_ipt_list),
        'task_opt': list(train_task_opt_list),
        'expl_ipt': list(train_expl_ipt_list),
        'expl_opt': list(train_expl_opt_list),
        'neg_op': list(train_expl_neg_op_list)
    }

    train_df = pd.DataFrame(mixup_unify_train_data, columns=['task_name', 'task_ipt', 'task_opt', 'expl_ipt', 'expl_opt', 'neg_op'])
    train_df.to_csv(to_train, encoding='utf-8')




def get_prompt_test_data(
        data_name='esnli',
        prompt_test='prompt.csv',
        to_test='prompt_expl_data_test.csv',
        select_mode='randomone'  # greedybec, greedywhy
):

    prompt_data = pd.read_csv(prompt_test)

    test_task_name = []
    test_task_ipt_list = []
    test_task_opt_list = []
    test_expl_ipt_list = []
    test_expl_opt_list = []

    # name_map = {
    #     'science_qa': '[SCIQA]',
    #     'aquarat_math': '[AQUAMATH]',
    #     'liar_plus': '[LIARPLUS]',
    #     'esnli': '[ESNLI]',
    #     'ecqa': '[ECQA]',
    #     'senmaking': '[senmaking]',
    #     'pubhealth': '[PUBHEALTH]',
    #     'winowhy': '[WINOWHY]',
    #     'e_delta_nli': '[EDELTANLI]'
    # }

    name_map_v2 = {
        'science_qa': 'This is a science exam question and answer.<|exp|>',
        'aquarat_math': 'This is an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'This is a journalistic claim and veracity label.<|exp|>',
        'esnli': 'This is a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'This is a commonsense question and answer.<|exp|>',
        'senmaking': 'There are two statements and select which one is true.<|exp|>',  #  makes sense
        'pubhealth': 'This is a public health claim and veracity label.<|exp|>',
        'winowhy': 'This is a statement and pronoun coreference resolution result.<|exp|>',
        'e_delta_nli': 'This is a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }

    name_map_v3 = {
        'science_qa': 'Let\'s explain a science exam question and answer.<|exp|>',
        'aquarat_math': 'Let\'s explain an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'Let\'s explain a journalistic claim and veracity label.<|exp|>',
        'esnli': 'Let\'s explain a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'Let\'s explain is a commonsense question and answer.<|exp|>',
        'senmaking': 'Let\'s explain two statements where only one statement is true.<|exp|>',  #  makes sense
        'pubhealth': 'Let\'s explain a public health claim and veracity label.<|exp|>',
        'winowhy': 'Let\'s explain a statement and pronoun coreference resolution result.<|exp|>',
        'e_delta_nli': 'Let\'s explain a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }


    count = 0
    special_token_x = name_map_v3[data_name]  # 拼在 解释句子前面
    special_token_y = name_map_v2[data_name]
    for index, row in prompt_data.iterrows():
        expl_ipt_list = row['exp_because'].split('\t') + row['exp_why'].split('\t')
        assert len(expl_ipt_list) == 6

        if select_mode == "randomone":
            tmp = random.choice(expl_ipt_list).strip().replace('\n', ' ')
        elif select_mode == "greedybec":
            tmp = expl_ipt_list[0]
        elif select_mode == "greedywhy":
            tmp = expl_ipt_list[3]
        elif select_mode == 'fews_cla':
            tmp = row['fews_cla']
            if isinstance(tmp, float):
                tmp = expl_ipt_list[0]
        else:
            tmp = '\t'.join(expl_ipt_list).replace('\t', '. ').replace('.. ', '. ').strip()
        # if tmp == '':
        #     tmp = '\t'.join(expl_ipt_list).replace('\t', '. ').replace('.. ', '. ').strip()  # use all
        #     if tmp == '':
        #         pdb.set_trace()

        tmp = special_token_x + tmp
        if tmp[-1].isalpha():  # 不是以标点符号结束
            tmp += '.'

        if data_name == 'esnli':
            s1 = row['Sentence1']
            s2 = row['Sentence2']
            label = row['gold_label']
            expl_opt = row['Explanation_1']

            if s1[-1].isalpha() is False:
                s1 = s1[:-1].strip()
            if s1.startswith('I ') is False:
                s1 = s1.lower()

            if s2[-1].isalpha() is False:
                s2 = s2[:-1].strip()
            if s2.startswith('I ') is False:
                s2 = s2.lower()

            task_ipt = special_token_y + 'Premise is ' + s1 + ' ' + 'Hypothesis is ' + s2
            task_opt = 'The label is ' + label
        else:  # ecqa
            que = row['q_text']
            ans = row['q_ans']

            expl_opt = row['taskA_pos']
            task_ipt = special_token_y + que
            task_opt = 'The answer is ' + ans

        test_task_name.append(data_name)
        test_task_ipt_list.append(task_ipt)
        test_task_opt_list.append(task_opt)
        test_expl_ipt_list.append(tmp)
        test_expl_opt_list.append(expl_opt)

        count += 1

    print('remove data name: ', data_name)
    print('test: ', count)

    # print('test size: ', len(test_task_name))
    # prompt_test_data = list(
    #     zip(test_task_ipt_list, test_task_opt_list, test_expl_ipt_list, test_expl_opt_list))
    # random.shuffle(prompt_test_data)
    # test_task_ipt_list, test_task_opt_list, test_expl_ipt_list, test_expl_opt_list = zip(
    #     *prompt_test_data)

    remove_unify_test_data = {
        'task_ipt': list(test_task_ipt_list),
        'task_opt': list(test_task_opt_list),
        'expl_ipt': list(test_expl_ipt_list),
        'expl_opt': list(test_expl_opt_list)
    }

    test_df = pd.DataFrame(remove_unify_test_data, columns=['task_ipt', 'task_opt', 'expl_ipt', 'expl_opt'])
    test_df.to_csv(to_test, encoding='utf-8')


def get_fse_data(
        data_name='esnli',
        prompt_test='prompt.csv',
        to_test='prompt_expl_data_test.csv',
        select_mode='randomone'  # greedybec, greedywhy
):

    prompt_data = pd.read_csv(prompt_test)

    test_task_name = []
    test_task_ipt_list = []
    test_task_opt_list = []
    test_expl_ipt_list = []
    test_expl_opt_list = []

    # name_map = {
    #     'science_qa': '[SCIQA]',
    #     'aquarat_math': '[AQUAMATH]',
    #     'liar_plus': '[LIARPLUS]',
    #     'esnli': '[ESNLI]',
    #     'ecqa': '[ECQA]',
    #     'senmaking': '[senmaking]',
    #     'pubhealth': '[PUBHEALTH]',
    #     'winowhy': '[WINOWHY]',
    #     'e_delta_nli': '[EDELTANLI]'
    # }

    name_map_v2 = {
        'science_qa': 'This is a science exam question and answer.<|exp|>',
        'aquarat_math': 'This is an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'This is a journalistic claim and veracity label.<|exp|>',
        'esnli': 'This is a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'This is a commonsense question and answer.<|exp|>',
        'senmaking': 'There are two statements and select which one is true.<|exp|>',  #  makes sense
        'pubhealth': 'This is a public health claim and veracity label.<|exp|>',
        'winowhy': 'This is a statement and pronoun coreference resolution result.<|exp|>',
        'e_delta_nli': 'This is a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }

    name_map_v3 = {
        'science_qa': 'Let\'s explain a science exam question and answer.<|exp|>',
        'aquarat_math': 'Let\'s explain an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'Let\'s explain a journalistic claim and veracity label.<|exp|>',
        'esnli': 'Let\'s explain a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'Let\'s explain is a commonsense question and answer.<|exp|>',
        'senmaking': 'Let\'s explain two statements where only one statement is true.<|exp|>',  #  makes sense
        'pubhealth': 'Let\'s explain a public health claim and veracity label.<|exp|>',
        'winowhy': 'Let\'s explain a statement and pronoun coreference resolution result.<|exp|>',
        'e_delta_nli': 'Let\'s explain a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }


    count = 0
    special_token_x = name_map_v3[data_name]  # 拼在 解释句子前面
    special_token_y = name_map_v2[data_name]
    for index, row in prompt_data.iterrows():
        tmp = row['predicted_explanation']

        tmp = special_token_x + tmp
        if tmp[-1].isalpha():  # 不是以标点符号结束
            tmp += '.'

        if data_name == 'esnli':
            s1 = row['Sentence1']
            s2 = row['Sentence2']
            label = row['gold_label']
            expl_opt = row['Explanation_1']

            if s1[-1].isalpha() is False:
                s1 = s1[:-1].strip()
            if s1.startswith('I ') is False:
                s1 = s1.lower()

            if s2[-1].isalpha() is False:
                s2 = s2[:-1].strip()
            if s2.startswith('I ') is False:
                s2 = s2.lower()

            task_ipt = special_token_y + 'Premise is ' + s1 + ' ' + 'Hypothesis is ' + s2
            task_opt = 'The label is ' + label
        else:  # ecqa
            que = row['question']
            ans = row['gold_label']

            expl_opt = row['gold_explanation']
            task_ipt = special_token_y + que
            task_opt = 'The answer is ' + ans

        test_task_name.append(data_name)
        test_task_ipt_list.append(task_ipt)
        test_task_opt_list.append(task_opt)
        test_expl_ipt_list.append(tmp)
        test_expl_opt_list.append(expl_opt)

        count += 1

    print('remove data name: ', data_name)
    print('test: ', count)

    # print('test size: ', len(test_task_name))
    # prompt_test_data = list(
    #     zip(test_task_ipt_list, test_task_opt_list, test_expl_ipt_list, test_expl_opt_list))
    # random.shuffle(prompt_test_data)
    # test_task_ipt_list, test_task_opt_list, test_expl_ipt_list, test_expl_opt_list = zip(
    #     *prompt_test_data)

    remove_unify_test_data = {
        'task_ipt': list(test_task_ipt_list),
        'task_opt': list(test_task_opt_list),
        'expl_ipt': list(test_expl_ipt_list),
        'expl_opt': list(test_expl_opt_list)
    }

    test_df = pd.DataFrame(remove_unify_test_data, columns=['task_ipt', 'task_opt', 'expl_ipt', 'expl_opt'])
    test_df.to_csv(to_test, encoding='utf-8')



if __name__ == '__main__':
    # jizhi
    # CODE_PATH='/apdcephfs/share_916081/qintongli/Explanation_code'
    # DIR_PATH='/apdcephfs/share_916081/qintongli/Explanation_code/NLEXP-datasets'
    # NEGATIVE_DIR_PATH='/apdcephfs/share_916081/qintongli/Explanation_code/process_data/get_negative_x'

    # hk
    CODE_PATH='/home/negative_unify_data0819qtli/explanation/Explanation_code'
    DIR_PATH='/home/qtli/explanation/Explanation_code/NLEXP-datasets'
    NEGATIVE_DIR_PATH='/home/qtli/explanation/Explanation_code/process_data/get_negative_x'

    unify_data_file = os.path.join(DIR_PATH, 'unify_expl_dataset.json')
    unify_data = json.load(open(unify_data_file, 'r'))

    ### read prompt data and get common sentences
    # ecqa_test = pd.read_csv(os.path.join(CODE_PATH, 'data/ECQA/new_exp_cqa_data_test.csv'))
    # esnli_test = pd.read_csv(os.path.join(CODE_PATH, 'data/ESNLI/small_test.csv'))
    # write_general_noise([ecqa_test, esnli_test], to_file=os.path.join(DIR_PATH, 'prompt_general_noise.json'))

    ### get common sentences from prompt data
    noise_dict, noise_list = get_general_noise(file=os.path.join(DIR_PATH, 'prompt_general_noise.json'))


    ### read external knowledge
    # write_avail_phrases(
    #     entity_file=os.path.join(DIR_PATH, "conceptnet_entity.csv"),
    #     negation_file=os.path.join(DIR_PATH, "negation.txt"),
    #     antonym_file=os.path.join(DIR_PATH, "conceptnet_antonym.txt"),
    #     to_file=os.path.join(DIR_PATH, "token_resources.json")
    # )

    ### get external knowledge
    avail_phrases, negation_word, antonym_word = get_avail_phrases(file=os.path.join(DIR_PATH, 'token_resources.json'))
    print('size of avail_phrases: {}, size of negation_word: {}, size of antonym_word: {}'.format(len(avail_phrases), len(negation_word), len(antonym_word)))

    ### read training set of each data and build vocabulary about pos and freq
    # write_unify_vocab(
    #     unify_data=unify_data,
    #     avail_phrases=avail_phrases,
    #     to_file=os.path.join(DIR_PATH, "unify_vocab.json")
    # )

    ### get freq vocabulary and pos vocabulary
    unify_vocab = get_unify_vocab(file=os.path.join(DIR_PATH, "unify_vocab.json"))
    print('unify_vocab: ', unify_vocab.keys())
    pos_vocab_entity = get_pos_vocab(unify_vocab)


    ## common negative ops 0809
    # type_dict = {"repeat": 0.2, "replace": 0.2, "shuffle": 0.2, "neg": 0.2, 'noise': 0.2}
    # type_list = list(type_dict.keys())
    # type_prob_list = []
    # for t in type_list:
    #     type_prob_list.append(type_dict[t])
    #
    # time_list = [1,2,3,4,5]  # 对于每个expl，负样本操作的次数候选列表以及对应概率
    # time_prob_list = [0.2,0.2,0.2,0.2,0.2]


    # further negative ops 0819
    # type_dict = {'para_infill': 0.7, 'sen_infill': 0.3}
    # type_list = list(type_dict.keys())
    # type_prob_list = []
    # for t in type_list:
    #     type_prob_list.append(type_dict[t])
    #
    # time_list = [1,2,3]  # 对于每个expl，负样本操作的次数候选列表以及对应概率
    # time_prob_list = [0.7,0.2,0.1]

    # 0906
    type_dict = {"repeat": 0.15, "replace": 0.1, "shuffle": 0.15, "neg": 0.1,
                 'para_infill': 0.25, 'sen_infill': 0.25}  # 'noise': 0.2,
    type_list = list(type_dict.keys())
    type_prob_list = []
    for t in type_list:
        type_prob_list.append(type_dict[t])

    time_list = [2,3,4]  # 对于每个expl，负样本操作的次数候选列表以及对应概率
    time_prob_list = [0.3,0.4,0.3]


    # similar_data_dict = json.load(open(os.path.join(NEGATIVE_DIR_PATH, 'contriever_results', 'merged_all_sentences.json'), 'r'))

    ### Setup glm
    # glm_model, glm_tokenizer, glm_args, glm_device = glm_setup()


    ### read negative ops and save
    # write_negatvie_data(
    #     unify_vocab=unify_vocab,
    #     unify_data=unify_data,
    #     type_list=type_list,
    #     time_list=time_list,
    #     time_prob_list=time_prob_list,
    #     type_prob_list=type_prob_list,
    #     antonym_word=antonym_word,
    #     pos_vocab_entity=pos_vocab_entity,
    #     negation_word=negation_word,
    #     avail_phrases=avail_phrases,
    #     noise_dict=noise_dict,
    #     noise_list=noise_list,
    #     similar_data_dict=similar_data_dict,
    #     # to_file=os.path.join(DIR_PATH,'negative_unify_data0819.json'),
    #     to_file=os.path.join(NEGATIVE_DIR_PATH, 'negative_unify_data0906.json'),
    #     max_size=100,
    #     glm_model=glm_model,
    #     glm_tokenizer=glm_tokenizer,
    #     glm_args=glm_args,
    #     glm_device=glm_device
    # )


    ### get processed negative data
    # negative_data = json.load(open(os.path.join(NEGATIVE_DIR_PATH, 'negative_unify_data0906.json'), 'r'))
    # print('negative_data: ', negative_data.keys())


    # #### read negative data and unify_data and combine them
    # combine_negative_data(
    #     unify_data=unify_data,
    #     negative_data=negative_data,
    #     to_file=os.path.join(NEGATIVE_DIR_PATH, 'unify_expl_dataset0913.json')
    # )

    # unify_data = json.load(open(os.path.join(NEGATIVE_DIR_PATH, 'unify_expl_dataset0810.json')))
    # unify_data = json.load(open(os.path.join(NEGATIVE_DIR_PATH, 'unify_expl_dataset0830.json')))
    # unify_data = json.load(open(os.path.join(NEGATIVE_DIR_PATH, 'unify_expl_dataset0906.json')))
    # unify_data = json.load(open(os.path.join(NEGATIVE_DIR_PATH, 'unify_expl_dataset0913.json')))

    ### add task special token before the explanation_ipt
    # mixup_unify_data(
    #     unify_data=unify_data,
    #     to_train=os.path.join(NEGATIVE_DIR_PATH, 'infill_unify_data/v3/unify_expl_data_train_woecqa.csv'),
    #     to_dev=os.path.join(NEGATIVE_DIR_PATH, 'infill_unify_data/v3/unify_expl_data_dev_woecqa.csv'),
    #     to_test=os.path.join(NEGATIVE_DIR_PATH, 'infill_unify_data/v3/unify_expl_data_test_woecqa.csv'),
    #     to_remove_test=os.path.join(NEGATIVE_DIR_PATH, 'infill_unify_data/v3/unify_expl_data_test_ecqa.csv'),
    #     remove_data_list=['ecqa']
    # )
    '''
    dev size:  15253
    test size:  15628
    train size:  152125
    '''

    # mixup_unify_data(
    #     unify_data=unify_data,
    #     to_train=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/unify_expl_data_train_woesnli.csv'),
    #     to_dev=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/unify_expl_data_dev_woesnli.csv'),
    #     to_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/unify_expl_data_test_woesnli.csv'),
    #     to_remove_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/unify_expl_data_test_esnli.csv'),
    #     remove_data_list=['esnli']
    # )
    '''
    dev size:  7989
    test size:  10400
    train size:  118680
    '''


    # select_mode = 'randomone'  # greedybec, greedywhy
    # select_mode = 'all'
    # get_prompt_test_data(
    #     data_name='esnli',
    #     prompt_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/prop_exp_esnli_data_test0913.csv'),
    #     to_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/{}_prompt_expl_data_test_esnli.csv'.format(select_mode)),
    #     select_mode=select_mode,
    # )
    #
    # get_prompt_test_data(
    #     data_name='ecqa',
    #     prompt_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/prop_exp_cqa_data_test0913.csv'),
    #     to_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/{}_prompt_expl_data_test_ecqa.csv'.format(select_mode)),
    #     select_mode=select_mode
    # )
    #
    #
    # get_fse_data(
    #     data_name='ecqa',
    #     prompt_test='/home/qtli/explanation/Explanation_code/data/few_shot_explanations/data/commonsenseqa_gpt3_generations/test_greedy_generations_conditioned_on_ecqa_prompts.csv',
    #     to_test='/home/qtli/explanation/Explanation_code/data/few_shot_explanations/data/commonsenseqa_gpt3_generations/{}_prompt_expl_data_test_ecqa.csv'.format(select_mode),
    #     select_mode=select_mode
    # )


    # combine fewshot-cla results with prompt data
    # df = pd.read_csv('~/explanation/Explanation_code/process_data/get_negative_x/common_unify_data/v3/prop_exp_cqa_data_test0913.csv')
    # fews_cla_result = json.load(open('/home/qtli/explanation/Explanation_code/code/fewshot-expl/ecqa_prompt_result.json'))
    # df['fews_cla'] = fews_cla_result
    # df.to_csv('~/explanation/Explanation_code/process_data/get_negative_x/common_unify_data/v3/prop_exp_cqa_data_test0913_full.csv')
    #
    # select_mode = 'fews_cla'
    # get_prompt_test_data(
    #     data_name='ecqa',
    #     prompt_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/prop_exp_cqa_data_test0913_full.csv'),
    #     to_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/{}_prompt_expl_data_test_ecqa.csv'.format(select_mode)),
    #     select_mode=select_mode
    # )

    # df = pd.read_csv('~/explanation/Explanation_code/process_data/get_negative_x/common_unify_data/v3/prop_exp_esnli_data_test0913.csv')
    # fews_cla_result = json.load(open('/home/qtli/explanation/Explanation_code/code/fewshot-expl/nli_prompt_result.json'))
    # df['fews_cla'] = fews_cla_result
    # df.to_csv('~/explanation/Explanation_code/process_data/get_negative_x/common_unify_data/v3/prop_exp_esnli_data_test0913_full.csv')

    select_mode = 'fews_cla'
    get_prompt_test_data(
        data_name='esnli',
        prompt_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/prop_exp_esnli_data_test0913_full.csv'),
        to_test=os.path.join(NEGATIVE_DIR_PATH, 'common_unify_data/v3/{}_prompt_expl_data_test_esnli.csv'.format(select_mode)),
        select_mode=select_mode
    )