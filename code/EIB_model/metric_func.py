import pdb
from collections import Counter
import collections
import math
from rouge import Rouge
import numpy as np
from nlgeval import NLGEval
from torch.nn import CrossEntropyLoss
import torch
from tqdm import tqdm
import bert_score
import sacrebleu
import json
from collections import defaultdict
from itertools import combinations


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts

def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def cal_bleu(target_list, gen_seqs):
    references = [[[x for x in p.strip(' ').split(' ')]] for p in target_list]
    predictions = [str(xs).strip(' ').split(' ') if str(xs).strip(' ') != '' else '' for xs in gen_seqs]

    ipt = Counter({'reference_corpus': references, 'translation_corpus': predictions})
    bleu1 = _compute_bleu(**ipt, max_order=1)
    bleu2 = _compute_bleu(**ipt, max_order=2)
    bleu4 = _compute_bleu(**ipt, max_order=4)
    print('bleu1: ', bleu1[0])
    print('bleu2: ', bleu2[0])
    print('bleu4: ', bleu4[0])

    '''
    In [1]: import sacrebleu
   ...: 
   ...: refs = [ # First set of references
   ...:          ['The dog bit the man.', 'It was not unexpected.', 'The man bit him first.'],
   ...:          # Second set of references
   ...:          ['The dog had bit the man.', 'No one was surprised.', 'The man had bitten the dog.'],
   ...:        ]
   ...: sys = ['The dog bit the man.', "It wasn't surprising.", 'The man had just bitten him.']

    In [2]: sacrebleu.corpus_bleu(sys, refs)
    Out[2]: BLEU = 48.53 82.4/50.0/45.5/37.5 (BP = 0.943 ratio = 0.944 hyp_len = 17 ref_len = 18)

    '''
    sbleu = sacrebleu.corpus_bleu(gen_seqs, target_list)
    print('sacrebleu: ', sbleu)
    return bleu1[0], bleu2[0], bleu4[0], sbleu


def _get_dist(res):
    if res == []:
        return 0, 0, 0, 0, 0
    unigrams = []
    bigrams = []
    avg_len = 0.
    ma_dist1, ma_dist2 = 0., 0.
    for q, r in enumerate(res):
        ugs = r
        bgs = []
        i = 0
        while i < len(ugs) - 1:
            bgs.append(ugs[i] + ugs[i + 1])
            i += 1
        unigrams += ugs
        bigrams += bgs
        ma_dist1 += len(set(ugs)) / (float)(len(ugs) + 1e-16)
        ma_dist2 += len(set(bgs)) / (float)(len(bgs) + 1e-16)
        avg_len += len(ugs)
    n = len(res)
    ma_dist1 /= n
    ma_dist2 /= n
    mi_dist1 = len(set(unigrams)) / (float)(len(unigrams))
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams)+1e-16)
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len


def cal_dist(gen_seqs):
    predictions = [str(xs).strip(' ').split(' ') if str(xs).strip(' ') != '' else '' for xs in gen_seqs]
    ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = _get_dist(predictions)
    print('dist1: ', mi_dist1 * 100)
    print('dist2: ', mi_dist2 * 100)
    return mi_dist1 * 100, mi_dist2 * 100


def cal_rouge(target_list, gen_seqs):
    '''
    from rouge import Rouge

    hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"

    reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"

    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)

    :param target_list:
    :param gen_seqs:
    :return:
    '''
    rouge = Rouge()
    scores = rouge.get_scores(target_list, gen_seqs, avg=True)  # {"rouge-1": {"f": _, "p": _, "r": _}, "rouge-2" : { ..     }, "rouge-l": { ... }}
    print('rouge: ', scores)


def cal_cider(references=None, hypothesis=None, hypo_file=None, ref_file=None):
    '''
    environment: NLGEVAL_DATA=/Users/liqintong/Downloads/nlg-eval-master/downloads
    :return:
    '''
    # metrics_dict = compute_metrics(hypothesis=hypo_file,
    #                                references=[ref_file])
    # print('CIDEr')
    # print(metrics_dict)
    nlgeval = NLGEval(metrics_to_omit=['EmbeddingAverageCosineSimilarity', 'EmbeddingAverageCosineSimilairty', 'VectorExtremaCosineSimilarity', 'GreedyMatchingScore'])  # loads the models
    metrics_dict = nlgeval.compute_metrics(references, hypothesis)
    print('CIDEr')
    print(metrics_dict)
    return metrics_dict


def cal_avglen(sentences):
    lens = []
    for idx, s in enumerate(sentences):
        lens.append(len(s.split()))
    l = np.array(lens).mean()
    print('avg lens: ', l)
    return l


def cal_novelty_corpus(task_samples, generations):

    all_sa_ugs = []
    all_sa_bgs = []
    all_ge_ugs = []
    all_ge_bgs = []
    for sample, gen in zip(task_samples, generations):
        sa_ugs = sample.split()
        sa_bgs = []
        i = 0
        while i < len(sa_ugs) - 1:
            sa_bgs.append(sa_ugs[i] + sa_ugs[i + 1])
            i += 1
        all_sa_ugs.extend(sa_ugs)
        all_sa_bgs.extend(sa_bgs)

        ge_ugs = gen.split()
        ge_bgs = []
        i = 0
        while i < len(ge_ugs) - 1:
            ge_bgs.append(ge_ugs[i] + ge_ugs[i+1])
            i += 1
        all_ge_bgs.extend(ge_bgs)
        all_ge_ugs.extend(ge_bgs)

    all_sa_ugs = set(all_sa_ugs)
    all_sa_bgs = set(all_sa_bgs)
    dedup_all_ge_ugs = set(all_ge_ugs)
    dedup_all_ge_bgs = set(all_ge_bgs)

    diff_ugs = 0
    diff_bgs = 0
    for gu in dedup_all_ge_ugs:
        if gu not in all_sa_ugs:
            diff_ugs += 1
    norm_diff_ugs = (diff_ugs + 1e-16) / (float)(len(all_ge_ugs) + 1e-16)

    for gb in dedup_all_ge_bgs:
        if gb not in all_sa_bgs:
            diff_bgs += 1
    norm_diff_bgs = (diff_bgs + 1e-16) / (float)(len(all_ge_bgs) + 1e-16)

    print('novelty 1 (corpus): ', norm_diff_ugs)
    print('novelty 2 (corpus): ', norm_diff_bgs)
    return norm_diff_ugs, norm_diff_bgs



def cal_novelty_compare_sample(task_samples, generations1, generations2):
    ma_nov1, ma_nov2 = 0., 0.
    listaaa = []
    count = 0
    for sample, gen1, gen2 in zip(task_samples, generations1, generations2):
        # sample
        #sa_ugs = sample.translate(str.maketrans('', '', string.punctuation)).strip().split()
        sa_ugs = sample.split()
        sa_bgs = []
        i = 0
        while i < len(sa_ugs) - 1:
            sa_bgs.append(sa_ugs[i] + sa_ugs[i + 1])
            i += 1

        # generation
        ge_ugs = gen1.split()
        ge_bgs = []
        i = 0
        while i < len(ge_ugs) - 1:
            ge_bgs.append(ge_ugs[i] + ge_ugs[i+1])
            i += 1

        diff_ugs = 0
        diff_bgs = 0
        for gu in ge_ugs:
            if gu not in sa_ugs:
                diff_ugs += 1
        norm_diff_ugs = (diff_ugs + 1e-16) / (float)(len(ge_ugs) + 1e-16)

        for gb in ge_bgs:
            if gb not in sa_bgs:
                diff_bgs += 1
        norm_diff_bgs = (diff_bgs + 1e-16) / (float)(len(ge_bgs) + 1e-16)

        ma_nov1 += norm_diff_ugs
        ma_nov2 += norm_diff_bgs



        ge_ugs = gen2.split()
        ge_bgs = []
        i = 0
        while i < len(ge_ugs) - 1:
            ge_bgs.append(ge_ugs[i] + ge_ugs[i+1])
            i += 1

        diff_ugs = 0
        diff_bgs = 0
        for gu in ge_ugs:
            if gu not in sa_ugs:
                diff_ugs += 1
        norm_diff_ugs2 = (diff_ugs + 1e-16) / (float)(len(ge_ugs) + 1e-16)

        if norm_diff_ugs>norm_diff_ugs2:
            pdb.set_trace()
            listaaa.append(count)

        count += 1

    n = len(task_samples)
    ma_nov1 /= n
    ma_nov2 /= n
    print('novelty 1 (sample avg): ', ma_nov1)
    print('novelty 2 (sample avg): ', ma_nov2)
    print('len count: ', len(listaaa))
    print('count: ', listaaa)
    return ma_nov1, ma_nov2



def cal_novelty_sample(task_samples, generations):
    ma_nov1, ma_nov2 = 0., 0.
    for sample, gen in zip(task_samples, generations):
        sa_ugs = sample.split()
        sa_bgs = []
        i = 0
        while i < len(sa_ugs) - 1:
            sa_bgs.append(sa_ugs[i] + sa_ugs[i + 1])
            i += 1

        ge_ugs = gen.split()
        ge_bgs = []
        i = 0
        while i < len(ge_ugs) - 1:
            ge_bgs.append(ge_ugs[i] + ge_ugs[i+1])
            i += 1

        diff_ugs = 0
        diff_bgs = 0
        for gu in ge_ugs:
            if gu not in sa_ugs:
                diff_ugs += 1
        norm_diff_ugs = (diff_ugs + 1e-16) / (float)(len(ge_ugs) + 1e-16)

        for gb in ge_bgs:
            if gb not in sa_bgs:
                diff_bgs += 1
        norm_diff_bgs = (diff_bgs + 1e-16) / (float)(len(ge_bgs) + 1e-16)

        ma_nov1 += norm_diff_ugs
        ma_nov2 += norm_diff_bgs


    n = len(task_samples)
    ma_nov1 /= n
    ma_nov2 /= n
    print('novelty 1 (sample avg): ', ma_nov1)
    print('novelty 2 (sample avg): ', ma_nov2)
    return ma_nov1, ma_nov2


def cal_novelty_wrt_source(task_samples, generations):
    # todo: new_ngram/source ngram

    ma_nov1, ma_nov2 = 0., 0.
    for sample, gen in zip(task_samples, generations):
        # sample
        #sa_ugs = sample.translate(str.maketrans('', '', string.punctuation)).strip().split()
        sa_ugs = sample.split()
        sa_bgs = []
        i = 0
        while i < len(sa_ugs) - 1:
            sa_bgs.append(sa_ugs[i] + sa_ugs[i + 1])
            i += 1

        # generation
        ge_ugs = gen.split()
        ge_bgs = []
        i = 0
        while i < len(ge_ugs) - 1:
            ge_bgs.append(ge_ugs[i] + ge_ugs[i+1])
            i += 1

        diff_ugs = 0
        diff_bgs = 0
        for gu in ge_ugs:
            if gu not in sa_ugs:
                diff_ugs += 1
        norm_diff_ugs = (diff_ugs + 1e-16) / (float)(len(ge_ugs) + 1e-16)

        for gb in ge_bgs:
            if gb not in sa_bgs:
                diff_bgs += 1
        norm_diff_bgs = (diff_bgs + 1e-16) / (float)(len(ge_bgs) + 1e-16)

        ma_nov1 += norm_diff_ugs
        ma_nov2 += norm_diff_bgs

    n = len(task_samples)
    ma_nov1 /= n
    ma_nov2 /= n
    print('novelty 1 (sample avg): ', ma_nov1)
    print('novelty 2 (sample avg): ', ma_nov2)
    return ma_nov1, ma_nov2


def cal_ppl_by_gpt2(inputs, model, tokenizer, device):
    ppls = []
    for sen in tqdm(inputs):
        ipt = tokenizer([sen], return_tensors='pt')
        ipt_ids = ipt['input_ids'].to(device)
        attention_mask = ipt['attention_mask'].to(device)
        labels = ipt['input_ids'].to(device)
        outputs = model(input_ids=ipt_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1]
        shift_logits = logits[:,:-1,:].contiguous()
        shift_labels = ipt['input_ids'][:, 1:].contiguous().to(device)

        # flatten the sentence
        loss_fuc = CrossEntropyLoss(ignore_index=0, reduction='none')
        loss = loss_fuc(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(1, -1)
        loss = loss.mean()
        ppl = torch.exp(loss).item()
        ppls.append(ppl)

    maxv = max(ppls)
    cleanedList = []
    for x in ppls:
        if str(x) != 'nan':
            cleanedList.append(x)
        else:
            print(x)
            cleanedList.append(maxv)
    print('ppl: ', np.array(cleanedList).mean())
    return ppls


def cal_bertscore(candidates, refs):
    (P, R, F), hash_code = bert_score.score(cands=candidates, refs=refs, model_type="roberta-large",
                                            num_layers=17, idf=True, return_hash=True, lang='en')
    f_score = F.mean().item()
    print('BERTScore: %.5f' % f_score)
    return f_score

def nominal_metric(a, b):
    return a != b


def interval_metric(a, b):
    return (a-b)**2


def ratio_metric(a, b):
    return ((a-b)/(a+b))**2


def krippendorff_alpha(data, metric=interval_metric, force_vecmath=False, convert_items=float, missing_items=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):

    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items

    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''

    # number of coders
    m = len(data)

    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)

    # convert input data to a dict of items
    units = {}
    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)

        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))
    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values

    if n == 0:
        raise ValueError("No items to compare.")

    np_metric = (np is not None) and ((metric in (interval_metric, nominal_metric, ratio_metric)) or force_vecmath)

    Do = 0.
    for grades in units.values():
        if np_metric:
            gr = np.asarray(grades)
            Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        else:
            Du = sum(metric(gi, gj) for gi in grades for gj in grades)
        Do += Du / float(len(grades) - 1)
    Do /= float(n)

    if Do == 0:
        return 1.


    De = 0.
    for g1 in units.values():
        if np_metric:
            d1 = np.asarray(g1)
            for g2 in units.values():
                De += sum(np.sum(metric(d1, gj)) for gj in g2)
        else:
            for g2 in units.values():
                De += sum(metric(gi, gj) for gi in g1 for gj in g2)
    De /= float(n*(n-1))
    return 1.-Do/De if (Do and De) else 1.



def overall_krippendorff_alpha(dataset='ecqa'):
    a1 = json.load(open(dataset + '-100/a1.json', 'r'))
    a2 = json.load(open(dataset + '-100/a2.json', 'r'))
    a3 = json.load(open(dataset + '-100/a3.json', 'r'))
    a4 = json.load(open(dataset + '-100/a4.json', 'r'))
    a5 = json.load(open(dataset + '-100/a5.json', 'r'))
    alphas = defaultdict(dict)

    for model_name in a1.keys():
        if model_name == 'IB-beta0.0001' or model_name == 'IB-beta0.001':
            continue
        print('model_name: ', model_name)
        a1_list = []
        a2_list = []
        a3_list = []
        a4_list = []
        a5_list = []
        for metric in a1[model_name]:
            # print('metric: ', metric)
            if metric == 'expl':
                break
            a1_list.extend(a1[model_name][metric])
            a2_list.extend(a2[model_name][metric])
            a3_list.extend(a3[model_name][metric])
            a4_list.extend(a4[model_name][metric])
            a5_list.extend(a5[model_name][metric])
        all_data = (
            a1_list,
            a2_list,
            a3_list,
            a4_list,
            a5_list
        )
        result = krippendorff_alpha(all_data, interval_metric)
        print("Krippendorff's alpha: %.3f" % result)
        alphas[model_name]['01234'] = round(result, 3)

        total = [a1_list, a2_list, a3_list, a4_list, a5_list]
        poss = list(combinations([0, 1, 2, 3, 4], 3))
        for p in poss:
            list_of_list = [total[p[0]], total[p[1]], total[p[2]]]
            print('p: ', p)
            result = krippendorff_alpha(list_of_list, interval_metric)
            print("Krippendorff's alpha: %.3f" % result)
            alphas[model_name][str(p)] = [round(result, 3)]
        json.dump(alphas, open(dataset + '-100/alphas.json', 'w'), indent=4)


def choose_most_consistent_annotation(dataset='ecqa'):
    alphas = json.load(open(dataset + '-100/alphas.json', 'r'))
    a_list = defaultdict(list)
    for model_name in alphas:
        for a in alphas[model_name]:
            a_list[a].append(alphas[model_name][a])
    max_k = 0
    best_a = ''
    for a in a_list.keys():
        score = np.array(a_list[a]).mean()
        if score > max_k:
            max_k = score
            best_a = a
    print(dataset)
    print('best_a: ', best_a)
    print('max_k: ', max_k)

    '''
    ecqa
    best_a:  (0,1,4)
    max_k:  0.3435


    esnli
    best_a:  (0, 2, 4)
    max_k:  0.38925
    '''