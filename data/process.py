import pdb
from tqdm import tqdm
import pandas as pd
import json
from nltk.corpus import stopwords
import nltk
pos_tag = nltk.pos_tag

from nltk.stem import WordNetLemmatizer
lemma = WordNetLemmatizer().lemmatize
import string
import math
import random
import numpy as np
import re
import os
import copy

from utils.infilling.fill_blank_glm import glm_setup, glm_generate_one_sample

function_word = [".", ",", "!", "?", " "]
all_num, antonym_num = 0, 0
failure_num = {'repeat': 0, 'replace': 0, 'shuffle': 0, 'neg': 0, 'noise': 0, 'para_infill': 0, 'sen_infill': 0}


def write_avail_phrases(
        entity_file="utils/conceptnet_entity.csv",
        negation_file="utils/negation.txt",
        antonym_file="utils/conceptnet_antonym.txt",
        to_file="utils/token_resources.json"
):
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

    return avail_phrases, negation_word, antonym_word



def write_unify_vocab(
        unify_data,
        avail_phrases,
        unify_vocab=None,
        to_file='unify_vocab.json'
):
    '''for each data, read data, build its vocab (introduce conceptnet) about pos and frequency'''
    for data_name in unify_data:
        if data_name not in unify_vocab:
            unify_vocab[data_name] = {}
        for split in ['train']:  # 'dev', 'test'
            split_data = unify_data[data_name][split]
            for i, expl in tqdm(enumerate(split_data['explanation']), total=len(split_data['explanation'])):
                expl = expl.strip('\n').strip(' ')
                expls = re.split('\\.|\t', expl)
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
                        if word_pos[0] in unify_vocab[data_name]:
                            unify_vocab[data_name][word_pos[0]]["number"] += 1
                            if word_pos[1] in unify_vocab[data_name][word_pos[0]]:
                                unify_vocab[data_name][word_pos[0]][word_pos[1]] += 1
                            else:
                                unify_vocab[data_name][word_pos[0]][word_pos[1]] = 1
                        else:
                            unify_vocab[data_name][word_pos[0]] = {word_pos[1]: 1, "number": 1}

        vocab_list = sorted(unify_vocab[data_name], key=lambda x: unify_vocab[data_name][x]["number"], reverse=True)
        for v in vocab_list:
            v_num = unify_vocab[data_name][v]['number']
            unify_vocab[data_name][v].pop('number')
            pos_dict = dict(sorted(unify_vocab[data_name][v].items(), key=lambda x:x[1], reverse=True))
            pos_dict.update({'number': v_num})
            unify_vocab[data_name][v] = pos_dict
        json.dump(unify_vocab, open(to_file, 'w'))
    return unify_vocab



def get_pos_vocab(unify_vocab):
    '''align with get_unify_vocab(), get pos and frequency'''
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
    expls = transfer_expl_to_expls(expl)

    idx = np.random.choice(
        np.arange(len(expls)),
        1 + int(len(expls)/2),
        replace=False
    ).tolist()

    s = min(idx)
    tmp_expl = copy.deepcopy(expls)
    for l in idx:
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
                idx = np.random.choice(np.arange(len(expls)))
                gram_num = np.random.choice(np.arange(5)[1:])
                split_sen = expls[idx].strip().split()
                pointer_st = np.random.choice(np.arange(len(split_sen)))
                pointer_ed = pointer_st + gram_num  # ngram
                if pointer_ed > len(split_sen):
                    pointer_ed = pointer_st
                    pointer_st = pointer_ed - gram_num
                    if pointer_st < 0:
                        continue
                    else:
                        flag = False
                        break
            except:
                continue
        if flag:
            failure_num['repeat'] += 1
            return copy.deepcopy(expls)

        sen1, sen2, sen3 = " ".join(split_sen[:pointer_st]), " ".join(split_sen[pointer_st:pointer_ed]), " ".join(split_sen[pointer_ed:])
        tmp_st = copy.deepcopy(expls)
        tmp_st[idx] = " ".join([sen1, sen2, sen2, sen3]).strip()

        return tmp_st


    expls = transfer_expl_to_expls(expl)

    for i in range(int(len(expls)/2)):
        expls = repeat_sen_gram(expls)

    new_expl = ' '.join(expls)
    return new_expl


def replace_sentence(expl, data):
    global failure_num
    expls = transfer_expl_to_expls(expl)
    tmp_expls = copy.deepcopy(expls)

    flag = True
    for _ in range(10):
        try:
            idxs = np.random.choice(
                np.arange(len(expls)),
                np.random.choice(np.arange(1, len(expls))),
                replace=False
            )
            replace_st_id = np.random.choice(np.arange(len(data)))
            for idx in idxs:
                tmp_expls[idx] = np.random.choice(transfer_expl_to_expls(data[replace_st_id]))
            flag = False
            break
        except:
            continue
    if flag:
        failure_num['replace'] += 1
        return copy.deepcopy(expl)

    new_expls = ' '.join(tmp_expls)
    return new_expls


def replace_word(expl, antonym_word, pos_vocab_entity, avail_phrases):
    global all_num, antonym_num
    global failure_num

    def replace_one_word(expls):
        antonym = False
        flag_in = True
        flag = True

        tmp_expls = copy.deepcopy(expls)

        for _ in range(100):
            tmp_expls = copy.deepcopy(expls)
            idx = np.random.choice(np.arange(len(expls)))
            split_sen = tmp_expls[idx].split()
            pos_split_sen = pos_tag(split_sen)

            avail_w_id = []
            for w_id, w in enumerate(split_sen):
                if w in avail_phrases and w not in function_word:
                    avail_w_id.append(w_id)
            if len(avail_w_id) == 0:
                continue

            word_id = np.random.choice(avail_w_id)
            if pos_split_sen[word_id][1] not in pos_vocab_entity:
                continue

            lemma_word = lemma(pos_split_sen[word_id][0], 'v' if pos_split_sen[word_id][1][0] == 'V' else 'n')
            if lemma_word in antonym_word:
                antonym = True
                replace_word = np.random.choice(antonym_word[lemma_word])
            else:
                antonym = False
                word_freq = pos_vocab_entity[pos_split_sen[word_id][1]]
                replace_word = ""
                for _ in range(10):
                    replace_word = np.random.choice(word_freq["word"], p=word_freq["freq"]/np.sum(word_freq["freq"]))
                    if replace_word != pos_split_sen[word_id][0]:
                        flag_in = False
                        break

                if flag_in:
                    replace_word = pos_split_sen[word_id][0]

            split_sen[word_id] = replace_word
            tmp_expls[idx] = " ".join(split_sen)
            if antonym or flag_in is False:
                flag = False
                break
            else:
                continue

        if flag:
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
        final_num = np.random.choice(np.arange(1, int(num*0.15+1)))
    except:
        final_num = 1

    for _ in range(final_num):
        expls, antonym, pos = replace_one_word(expls)
        all_num += 1
        if antonym:
            antonym_num += 1

    new_expls = ' '.join(expls)
    return new_expls


def shuffle_sentence(expl, n_sentence=None):
    global failure_num

    def exchange(l, ids, target_ids):
        tmp_l = copy.deepcopy(l)
        for o_id, t_id in zip(ids, target_ids):
            tmp_l[o_id] = copy.deepcopy(l[t_id])
        return tmp_l

    expls = transfer_expl_to_expls(expl)
    if n_sentence is None:
        n_sentence = int(np.random.choice(np.arange(1, len(expls)+1)))

    flag = True
    tmp_expls = []
    for _ in range(10):
        sen_ids = np.random.choice(np.arange(len(expls)), n_sentence, replace=False)
        target_ids = np.random.permutation(sen_ids)
        tmp_expls = exchange(expls, sen_ids, target_ids)
        if expls != tmp_expls:
            flag = False
            break

    if flag:
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
        if n in negation_word:
            del sen[i]
            return pro(sen)

    neg_list = ["not", "n't"]
    for i, n in enumerate(sen):
        if n in ["would", "will", "can", "could", "may", "might", "shall", "should", "do", "does", "did", "am", "is", "are", "was", "were", "be", "been"]:
            sen.insert(i+1, np.random.choice(neg_list))
            return pro(sen)

    pos_sen = pos_tag(sen)
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
    '''add or delete negation words'''
    global failure_num
    flag = True

    expls = transfer_expl_to_expls(expl)
    tmp_expls = copy.deepcopy(expls)

    for _ in range(10):
        try:
            tmp_expls = copy.deepcopy(expls)
            idxs = np.random.choice(np.arange(len(expls)), np.random.choice(np.arange(1, len(expls)+1)), replace=False)
            for idx in idxs:
                if expls[idx].isalpha() is False:
                    continue
                tmp_expls_neg = change_neg_helper(expls[idx], negation_word)
                if tmp_expls_neg is not None:
                    tmp_expls[idx] = tmp_expls_neg
                    flag = False
            if flag == False:
                break
        except:
            continue

    if flag:
        failure_num['neg'] += 1
        return copy.deepcopy(expl)

    new_expl = ' '.join(tmp_expls)
    return new_expl


def paragraph_infilling(expl, similar_data_dict, before_num=None, after_num=None, infill_maxnum=2):
    '''
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
        np.arange(len(expls)),
        1 + int(len(expls)/3),
        replace=False
    ).tolist()
    idx.sort(reverse=False)

    tmp_expl = copy.deepcopy(expls)
    offset = 0
    for i, l in enumerate(idx):
        if before_num is None:
            before_num = random.choice(np.arange(infill_maxnum)) + 1
        if after_num is None:
            after_num = random.choice(np.arange(infill_maxnum)) + 1
        if similar_data_dict[expls[l]] == []:
            continue
        if random.random() < 0.5:
            to_infill_sentences = np.random.choice(
                similar_data_dict[expls[l]],
                before_num,
            )
            for j, s in enumerate(to_infill_sentences):
                tmp_expl.insert(l+offset, s['sentence'])
                offset += 1
        else:
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
    global failure_num

    expls = transfer_expl_to_expls(expl)
    if len(expls) == 0:
        failure_num['sen_infill'] += 1
        return expl
    tmp_expls = copy.deepcopy(expls)

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
                tmp_expls[idx] = process_trim_decode_tokens
            flag = False
            break
        except:
            continue
    if flag:
        failure_num['sen_infill'] += 1
        return copy.deepcopy(expl)

    new_expls = ' '.join(tmp_expls)
    return new_expls


def construct_mixexpl(
        unify_data,
        type_list,
        time_list,
        time_prob_list,
        type_prob_list,
        antonym_word,
        pos_vocab_entity,
        negation_word,
        avail_phrases,
        similar_data_dict,
        to_file=None,
        glm_model=None,
        glm_tokenizer=None,
        glm_args=None,
        glm_device=None
):
    global all_num, antonym_num, failure_num
    negative_unify_data = {}

    for data_name in unify_data:
        negative_unify_data[data_name] = {}

        for split in ['train', 'dev', 'test']:
            negative_unify_data[data_name][split] = []
            if split == 'train':
                max_num = 1500
            else:
                max_num = 200
            split_data = unify_data[data_name][split]['explanation'][:max_num]

            for idx, expl in enumerate(tqdm(split_data)):
                expl = str(expl)
                if expl.strip() == '' or expl == 'nan':
                    negative_unify_data[data_name][split].append({
                        'explanation': str(expl),
                        'neg_explanation': [],
                        'neg_op': []
                    })
                    continue

                chaotic_list = np.random.choice(
                    type_list,
                    np.random.choice(time_list, p=time_prob_list),
                    p=type_prob_list,
                ).tolist()
                nelist = []
                neoplist = []

                for c in chaotic_list:
                    new_expl = ''
                    if c == "repeat":   # we need save each negative operation name (c) in neoplist
                        if random.random() < 0.7:
                            new_expl = repeat_sentence(expl)
                            c += '_sentence'
                        else:
                            new_expl = repeat_ngram(expl)
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
                    elif c == "para_infill":
                        new_expl = paragraph_infilling(expl, similar_data_dict=similar_data_dict[data_name])   # use the off-the-shelf retrieved results by contriever
                    elif c == "sen_infill":
                        new_expl = sentence_infilling(expl, glm_model, glm_tokenizer, glm_args, glm_device)   # use off-the-shelf pretrained model

                    nelist.append(new_expl)
                    neoplist.append(c)

                negative_unify_data[data_name][split].append({
                    'explanation': str(expl),
                    'neg_explanation': nelist,
                    'neg_op': neoplist
                })
        json.dump(negative_unify_data, open(to_file, 'w'), indent=4)

    json.dump(negative_unify_data, open(to_file, 'w'), indent=4)

    return negative_unify_data





def combine_negative_data(
        unify_data,
        negative_data,
        to_file
):
    '''
    :param unify_data: original unify data
    :param negative_data: includes explanation (string) and neg_explanation (list)
    :param to_file:
    :return:
    '''
    for data_name in negative_data:
        print('data_name: ', data_name)
        for split in negative_data[data_name]:
            print('split: ', split)
            tmp = copy.deepcopy(unify_data[data_name][split])
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
                    neg_ipt = item['neg_explanation']
                    new_explanation_ipt.append(neg_ipt)
                    new_explanation_opt.append(opt)
                    if 'neg_op' not in item:
                        new_explanation_neg_op.append(item['neg_ops'])
                    else:
                        new_explanation_neg_op.append(item['neg_op'])
                else:
                    opts = opt.split('\t')
                    new_opts = []
                    for o in opts:
                        if o[-1].isalpha():
                            o += '.'
                        new_opts.append(o)
                    for o in new_opts:
                        new_task_ipt.append(task_ipt[idx])
                        new_task_opt.append(task_opt[idx])
                        neg_ipt = item['neg_explanation']
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
    return unify_data



def balance_noise_ratio(data):

    def del_some_noise(neg_exp, good_exp):
        len_r = len(neg_exp.split()) / len(good_exp.split())
        noise_list = []
        noise_sen = []
        subnegexp = re.split('([,.?;!])', neg_exp)
        subnegexp.append("")
        subnegexp = ["".join(i) for i in zip(subnegexp[0::2], subnegexp[1::2])]

        neg_exp_remove_punc = neg_exp.translate(str.maketrans('', '', string.punctuation)).split()
        good_exp_remove_punc = good_exp.translate(str.maketrans('', '', string.punctuation)).split()
        delete_noise_fail = 0

        if len_r > 2:
            for si, sen in enumerate(subnegexp):
                if sen == '': continue
                if sen[:-1].strip() not in good_exp and len(sen) != 1:
                    noise_sen.append(sen)

                    sen_remove_punc = sen.translate(str.maketrans('', '', string.punctuation)).split()
                    sen_c = len(sen_remove_punc) - 3
                    exist_good = False
                    if sen_c != 0:
                        for i in range(sen_c):
                            check = ' '.join(sen_remove_punc[i:i + 4])
                            if check in good_exp:
                                exist_good = True
                                break
                    if exist_good is False:
                        noise_list.append(sen)
            new_neg_exp = subnegexp
            nn_neg_exp = []
            for item in new_neg_exp:
                if len(item) > 1:
                    nn_neg_exp.append(item)
            new_neg_exp = copy.deepcopy(nn_neg_exp)
            noise_count = len(noise_list)

            if len(noise_list) == 0:
                while_c = 0
                while True:
                    while_c += 1
                    if while_c > 10:
                        new_neg_exp = "FAIL"
                        delete_noise_fail += 1
                        break
                    n = random.choice(new_neg_exp)
                    new_neg_exp.remove(n)
                    if len(''.join(new_neg_exp).split()) / len(good_exp.split()) <= 2:
                        new_neg_exp = ''.join(new_neg_exp)
                        break
            else:
                while_c = 0
                while True:
                    while_c += 1
                    if while_c > noise_count - 1:
                        if new_neg_exp == nn_neg_exp:
                            new_neg_exp = "FAIL"
                            delete_noise_fail += 1
                            break
                        else:
                            while len(''.join(new_neg_exp).split()) / len(good_exp.split()) > 3:
                                n = random.choice(new_neg_exp)
                                new_neg_exp.remove(n)
                            new_neg_exp = ''.join(new_neg_exp)
                            break
                    if len(noise_list) == 0:
                        new_neg_exp = ''.join(new_neg_exp)
                        break
                    n = random.choice(noise_list)
                    new_neg_exp.remove(n)
                    noise_list.remove(n)
                    if len(''.join(new_neg_exp).split()) / len(good_exp.split()) <= 2:
                        new_neg_exp = ''.join(new_neg_exp)
                        break
        else:
            new_neg_exp = neg_exp

        new_neg_exp = new_neg_exp.strip()
        return new_neg_exp

    def add_some_good(neg_exp, good_exp, good_4gram_candidates):
        subgoodexp = re.split('([,.?;!])', good_exp)
        subgoodexp.append("")
        subnegexp = re.split('([,.?;!])', neg_exp)
        subnegexp.append("")
        new_neg_exp = neg_exp
        for idx, sg in enumerate(good_4gram_candidates):
            new_neg_exp += (' ' + sg)
            if len(new_neg_exp.split()) / len(good_exp.split()) > 2.5:
                break
        new_neg_exp = new_neg_exp.strip()
        return new_neg_exp

    def good_keep(good_c, good_exp_remove_punc, good_exp, neg_exp):
        subgoodexp = re.split('([,.?;!])', good_exp)
        subgoodexp.append("")
        subgoodexp = ["".join(i) for i in zip(subgoodexp[0::2], subgoodexp[1::2])]

        count = 0
        ngram_list = []
        for i in range(good_c):
            check = ' '.join(good_exp_remove_punc[i:i + 4])
            if check in neg_exp:
                count += 1
            else:
                ngram_list.append(check)
        keep_r = count / good_c

        sort_dict = {}
        if keep_r < 0.5:
            for sg in subgoodexp:
                sg = sg.strip()
                sg_remove_punc = sg.translate(str.maketrans('', '', string.punctuation)).split()
                sort_dict[sg] = 0
                c = len(sg_remove_punc) - 3
                if c != 0:
                    count = 0
                    for i in range(c):
                        check = ' '.join(sg_remove_punc[i:i + 4])
                        if check in ngram_list:
                            count += 1
                    sort_dict[sg] = count
        new_sort_dict = dict(sorted(sort_dict.items(), key=lambda x: x[1], reverse=True))
        return keep_r, new_sort_dict

    def neg_add(neg_c, neg_exp_remove_punc, good_exp):
        count = 0
        noise_list = []
        to_delete = 0
        for i in range(neg_c):
            check = ' '.join(neg_exp_remove_punc[i:i + 4])
            if check not in good_exp:
                count += 1
                noise_list.append(check)
        noise_r = count / neg_c
        if noise_r < 0.5:
            to_delete = 0
        else:
            to_delete = math.ceil(count - neg_c * 0.5)
        return noise_r, noise_list, to_delete

    keep, noise, new_neg_ipt = [], [], []
    for index, row in data.iterrows():
        neg_exp = row['expl_ipt'].split('<|exp|>')[1].lower().strip()
        good_exp = row['expl_opt'].lower().strip()
        if neg_exp.translate(str.maketrans('', '', string.punctuation)).strip() == "":
            neg_exp = good_exp
        if neg_exp == good_exp:
            keep.append(1)
            noise.append(0)
            new_neg_ipt.append('SAME')
        else:
            all_ocur = True
            subgoodexp = re.split('[,.?;!]', good_exp)
            for sen in subgoodexp:
                sen = sen.strip()
                if sen == '': continue
                sen = sen.translate(str.maketrans('', '', string.punctuation))
                if sen not in neg_exp:
                    all_ocur = False
            if all_ocur == True:
                keep.append(1)
                noise.append(0)
                new_neg_ipt.append('SAME SHUFFLE')
            else:
                neg_exp_remove_punc = neg_exp.translate(str.maketrans('', '', string.punctuation)).split()
                neg_c = len(neg_exp_remove_punc) - 3  # check 4-gram
                good_exp_remove_punc = good_exp.translate(str.maketrans('', '', string.punctuation)).split()
                good_c = len(good_exp_remove_punc) - 3

                if neg_c == 0 or good_c == 0:
                    new_neg_ipt.append("NO CHANGE")
                else:
                    keep_r, ngram_dict = good_keep(good_c, good_exp_remove_punc, good_exp, neg_exp)
                    noise_r, noise_list, to_delete = neg_add(neg_c, neg_exp_remove_punc, good_exp)
                    keep.append(keep_r)
                    noise.append(noise_r)

                    if noise_r > 0.5:
                        new_neg_exp = del_some_noise(neg_exp, good_exp)
                        neg_exp = new_neg_exp
                    keep_r, ngram_dict = good_keep(good_c, good_exp_remove_punc, good_exp, neg_exp)
                    if keep_r < 0.5:
                        new_neg_exp = add_some_good(neg_exp, good_exp, good_4gram_candidates=ngram_dict)
                        neg_exp = new_neg_exp
                    new_neg_ipt.append(row['expl_ipt'].split('<|exp|>')[0] + '<|exp|>' + neg_exp.strip())
    data['expl_ipt'] = new_neg_ipt

    new_neg = []
    for row in data.iterrows():
        if row['new_expl_ipt'] == "SAME SHUFFLE":
            good = row['expl_opt']
            subgoodexp = re.split('([,.?;!])', good)
            subgoodexp.append("")
            subgoodexp = ["".join(i) for i in zip(subgoodexp[0::2], subgoodexp[1::2])]
            if len(subgoodexp) == 0:
                new_neg.append("EMPTY")
                continue
            if len(subgoodexp) == 1:
                if random.random() < 0.5:
                    new_expl = paragraph_infilling(good, similar_data_dict=similar_data_dict[
                        row['task_name']])
                else:
                    new_expl = sentence_infilling(subgoodexp, glm_model, glm_tokenizer, glm_args, glm_device)
                    if new_expl == good:
                        new_expl = "SEN FAIL"
                new_neg.append(new_expl)
            else:
                new_good = good
                if random.random() < 0.5:
                    new_expl = paragraph_infilling(new_good, similar_data_dict=similar_data_dict[
                        row['task_name']])
                else:
                    new_expl = sentence_infilling(subgoodexp, glm_model, glm_tokenizer, glm_args, glm_device)
                new_neg.append(new_expl)
        else:
            new_neg.append(row['new_expl_ipt'])
    data['expl_ipt'] = new_neg

    new = []
    for index, row in data.iterrows():
        neg_exp = row['new_expl_ipt_2']
        ori_neg_exp = row['expl_ipt'].split('<|exp|>')[1].strip()
        if isinstance(neg_exp, float):
            neg_exp = row['expl_ipt']
            new.append(neg_exp)
            continue
        if neg_exp.isupper():
            good_exp = row['expl_opt']
            new_neg_exp = del_some_noise(ori_neg_exp, good_exp).strip()
            neg_exp = row['expl_ipt'].split('<|exp|>')[0].strip() + '<|exp|>' + new_neg_exp
        else:
            if '<|exp|>' not in neg_exp:
                if 'Let\'s explain' in neg_exp:
                    pdb.set_trace()
                neg_exp = row['expl_ipt'].split('<|exp|>')[0].strip() + '<|exp|>' + neg_exp
        neg_exp = neg_exp.replace('[mask]', '').replace('[MASK]', '')
        new.append(neg_exp)
    data['expl_ipt'] = new


    new = []
    for index, row in data.iterrows():
        n = row['new_expl_ipt_3']
        ori_neg_exp = row['expl_ipt'].split('<|exp|>')[1].strip()
        good_exp = row['expl_opt']
        if n.split('<|exp|>')[1].strip() == '':
            new_neg_exp = del_some_noise(ori_neg_exp, good_exp).strip()
            new.append(n.split('<|exp|>')[0] + '<|exp|>' + new_neg_exp)
        elif '<|exp|>FAIL' in n:
            new.append(row['expl_ipt'])
        else:
            new.append(n)
    data['expl_ipt'] = new

    return data





def mixup_unify_data(
        unify_data,
        to_train='unify_expl_data_train.csv',
        to_dev='unify_expl_data_dev.csv',
        to_test='unify_expl_data_test.csv',
        name_map_y=None,
        name_map_x=None,
        name_map=None
):

    for data_name in unify_data:
        print('data_name: ', data_name)
        train_data = unify_data[data_name]['train']
        dev_data = unify_data[data_name]['dev']
        test_data = unify_data[data_name]['test']
        special_token_x = name_map_x[data_name]
        special_token_y = name_map_y[data_name]
        special_token = name_map[data_name]

        split_data = [train_data, dev_data, test_data]
        split_name = ['train', 'dev', 'test']

        for s_name, split in zip(split_name, split_data):
            task_name = []
            task_ipt_list = []
            task_opt_list = []
            expl_ipt_list = []
            expl_opt_list = []
            expl_neg_op_list = []

            count = 0
            for idx, items in enumerate(split['explanation_ipt']):
                if isinstance(items, str):
                    items = items.replace(' .', '.').replace('\n', '. ')
                    if items == split['explanation_opt'][idx]:
                        continue
                    tmp = special_token_x + items
                    tmp = tmp.replace('\t', '. ').replace('.. ', '. ').strip()
                    if tmp[-1].isalpha():
                        tmp += '.'
                    if (len(tmp.split()) + len(split['explanation_opt'][idx].split())) > 250 or len(train_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                        continue
                    tmp_opt = split['explanation_opt'][idx].replace('\t', '. ').replace('.. ', '. ').strip()
                    if tmp_opt[-1].isalpha():
                        tmp_opt += '.'

                    task_name.append(data_name)
                    expl_ipt_list.append(tmp)
                    expl_opt_list.append(tmp_opt)
                    tempt = split['task_ipt'][idx].replace(special_token, special_token_y)
                    task_ipt_list.append(tempt)
                    task_opt_list.append(split['task_opt'][idx])
                    count += 1

                else:
                    for jdx, item in enumerate(items):
                        item = item.replace(' .', '.').replace('\n', '. ').replace('[MASK]', '').replace('[mask]', '')
                        if item == split['explanation_opt'][idx]:
                            continue
                        tmp = special_token_x + item
                        if (len(tmp.split()) + len(split['explanation_opt'][idx].split())) > 250 or len(train_data['explanation_opt'][idx].split())<3 or len(tmp.split())<3:
                            continue

                        task_name.append(data_name)
                        expl_ipt_list.append(tmp)
                        expl_opt_list.append(split['explanation_opt'][idx])
                        tempt = split['task_ipt'][idx].replace(special_token, special_token_y)
                        task_ipt_list.append(tempt)
                        task_opt_list.append(split['task_opt'][idx])
                        expl_neg_op_list.append(split['explanation_neg_op'][idx])
                        count += 1
            print('{}: {}'.format(s_name, count))

            unify_split = list(zip(task_name, task_ipt_list, task_opt_list, expl_ipt_list, expl_opt_list, expl_neg_op_list))
            random.shuffle(unify_split)
            task_name, task_ipt_list, task_opt_list, expl_ipt_list, expl_opt_list, expl_neg_op_list = zip(*unify_split)

            mixup_unify_split_data = {
                'task_name': list(task_name),
                'task_ipt': list(task_ipt_list),
                'task_opt': list(task_opt_list),
                'expl_ipt': list(expl_ipt_list),
                'expl_opt': list(expl_opt_list),
                'neg_op': list(expl_neg_op_list)
            }

            split_df = pd.DataFrame(mixup_unify_split_data, columns=['task_name', 'task_ipt', 'task_opt', 'expl_ipt', 'expl_opt', 'neg_op'])

            split_df = balance_noise_ratio(split_df)

            if s_name == 'train':
                split_df.to_csv(to_train, encoding='utf-8')
            elif s_name == 'dev':
                split_df.to_csv(to_dev, encoding='utf-8')
            else:
                split_df.to_csv(to_test, encoding='utf-8')



def get_prompt_test_data(
        data_name='esnli',
        prompt_test='prompt.csv',
        to_test='prompt_expl_data_test.csv',
        select_mode='randomone',  # prompt, prompt_filter
        prompt_prefix='bec',  # bec why
        name_map_y=None,
        name_map_x=None
):
    select_mode = select_mode + '_' + prompt_prefix
    prompt_data = pd.read_csv(prompt_test)

    test_task_name = []
    test_task_ipt_list = []
    test_task_opt_list = []
    test_expl_ipt_list = []
    test_expl_opt_list = []


    count = 0
    special_token_x = name_map_x[data_name]
    special_token_y = name_map_y[data_name]
    for index, row in prompt_data.iterrows():
        expl_ipt_list = row['exp_because'].split('\t') + row['exp_why'].split('\t')
        assert len(expl_ipt_list) == 6

        if select_mode == "randomone":
            tmp = random.choice(expl_ipt_list).strip().replace('\n', ' ')
        elif select_mode == "prompt_bec":
            tmp = expl_ipt_list[0]
        elif select_mode == "prompt_why":
            tmp = expl_ipt_list[3]
        elif 'prompt_filter' in select_mode:
            tmp = row['fews_cla']
            if isinstance(tmp, float):
                tmp = expl_ipt_list[0]
        else:
            tmp = '\t'.join(expl_ipt_list).replace('\t', '. ').replace('.. ', '. ').strip()

        tmp = special_token_x + tmp
        if tmp[-1].isalpha():
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
        else:
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


    remove_unify_test_data = {
        'task_ipt': list(test_task_ipt_list),
        'task_opt': list(test_task_opt_list),
        'expl_ipt': list(test_expl_ipt_list),
        'expl_opt': list(test_expl_opt_list)
    }

    test_df = pd.DataFrame(remove_unify_test_data, columns=['task_ipt', 'task_opt', 'expl_ipt', 'expl_opt'])
    test_df.to_csv(to_test, encoding='utf-8')



if __name__ == '__main__':
    DIR_PATH = 'utils'
    unify_data = json.load(open('explanation_datasets/unify_expl_dataset.json', 'r'))

    ### read external knowledge
    avail_phrases, negation_word, antonym_word = write_avail_phrases(
        entity_file=os.path.join(DIR_PATH, "conceptnet_entity.csv"),
        negation_file=os.path.join(DIR_PATH, "negation.txt"),
        antonym_file=os.path.join(DIR_PATH, "conceptnet_antonym.txt"),
        to_file=os.path.join(DIR_PATH, "token_resources.json")
    )
    print('size of avail_phrases: {}, size of negation_word: {}, size of antonym_word: {}'.format(len(avail_phrases), len(negation_word), len(antonym_word)))


    ### read training set of each data and build vocabulary about pos and freq
    unify_vocab = write_unify_vocab(
        unify_data=unify_data,
        avail_phrases=avail_phrases,
        to_file=os.path.join(DIR_PATH, "unify_vocab.json")
    )
    print('unify_vocab: ', unify_vocab.keys())

    pos_vocab_entity = get_pos_vocab(unify_vocab)


    type_dict = {"repeat": 0.15, "replace": 0.1, "shuffle": 0.15, "neg": 0.1,
                 'para_infill': 0.25, 'sen_infill': 0.25}
    type_list = list(type_dict.keys())
    type_prob_list = []
    for t in type_list:
        type_prob_list.append(type_dict[t])

    time_list = [2,3,4]  # for each explanation, times and frequencies of the negation operations
    time_prob_list = [0.3,0.4,0.3]

    similar_data_dict = json.load(open('utils/contriever_src/contriever_results/merged_all_sentences.json', 'r'))

    ### Setup glm
    glm_model, glm_tokenizer, glm_args, glm_device = glm_setup()

    ### read negative ops and save
    prop_unify_data = construct_mixexpl(
        unify_data=unify_data,
        type_list=type_list,
        time_list=time_list,
        time_prob_list=time_prob_list,
        type_prob_list=type_prob_list,
        antonym_word=antonym_word,
        pos_vocab_entity=pos_vocab_entity,
        negation_word=negation_word,
        avail_phrases=avail_phrases,
        similar_data_dict=similar_data_dict,
        to_file='MixExpl/MixExpl_neg.json',
        glm_model=glm_model,
        glm_tokenizer=glm_tokenizer,
        glm_args=glm_args,
        glm_device=glm_device
    )


    # #### read negative data and unify_data and combine them
    mixexpl_data = combine_negative_data(
        unify_data=unify_data,
        negative_data=prop_unify_data,
        to_file='MixExpl/MixExpl.json'
    )


    name_map_y = {
        'science_qa': 'This is a science exam question and answer.<|exp|>',
        'aquarat_math': 'This is an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'This is a journalistic claim and veracity label.<|exp|>',
        'esnli': 'This is a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'This is a commonsense question and answer.<|exp|>',
        'senmaking': 'There are two statements and select which one is true.<|exp|>',  #  makes sense
        'pubhealth': 'This is a public health claim and veracity label.<|exp|>',
        'e_delta_nli': 'This is a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }

    name_map_x = {
        'science_qa': 'Let\'s explain a science exam question and answer.<|exp|>',
        'aquarat_math': 'Let\'s explain an algebraic word problem and solving.<|exp|>',
        'liar_plus': 'Let\'s explain a journalistic claim and veracity label.<|exp|>',
        'esnli': 'Let\'s explain a premise, hypothesis, and relation label between premise and hypothesis.<|exp|>',
        'ecqa': 'Let\'s explain is a commonsense question and answer.<|exp|>',
        'senmaking': 'Let\'s explain two statements where only one statement is true.<|exp|>',  #  makes sense
        'pubhealth': 'Let\'s explain a public health claim and veracity label.<|exp|>',
        'e_delta_nli': 'Let\'s explain a premise, hypothesis, update, and a label about whether the update weakens or strengthens the entailment of the hypothesis by the premise.<|exp|>'
    }

    name_map = {
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


    ### add task special token before the explanation_ipt
    mixup_unify_data(
        unify_data=unify_data,
        to_train='MixExpl/train.csv',
        to_dev='MixExpl/dev.csv',
        to_test='MixExpl/test.csv',
        name_map_y=name_map_y,
        name_map_x=name_map_x,
        name_map=name_map
    )


    for select_mode in ['prompt', 'prompt_filter']:
        get_prompt_test_data(
            data_name='esnli',
            prompt_test='explanation_datasets/esnli/esnli_explanation_cands.csv',
            to_test='MixExpl/{}_esnli_test.csv'.format(select_mode),
            select_mode=select_mode,
            name_map_y=name_map_y,
            name_map_x=name_map_x
        )

        get_prompt_test_data(
            data_name='ecqa',
            prompt_test='explanation_datasets/ecqa/ecqa_explanation_cands.csv',
            to_test='MixExpl/{}_ecqa_test.csv'.format(select_mode),
            select_mode=select_mode,
            name_map_y=name_map_y,
            name_map_x=name_map_x
        )



