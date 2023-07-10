import json
import logging
logger = logging.getLogger()
import os
import collections
import math
import argparse
from torchstat import stat  # only one input
from thop import profile


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", default='ecqa', type=str, required=True, help="ecqa or esnli")
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The path to the input training data file (a text file).")
    parser.add_argument("--train_data_file", default="ed_train_prop.json", type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--dev_data_file", default="ed_dev_prop.json", type=str,
                        help="The input dev data file (a text file).")
    parser.add_argument("--test_data_file", default="ed_test_prop.json", type=str,
                        help="The input test data file (a text file).")
    parser.add_argument("--mixexpl_task_name", default="taskname", type=str,
                        help="for testing performance of separate task which is included in MixExpl corpus")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--save_every", action='store_true', help="whether save the last epoch")


    parser.add_argument('--use_wandb', action='store_true', help='whether use wandb')
    parser.add_argument("--task_type", default="IB", type=str, help="downstream task type ...")

    parser.add_argument("--model_type", default="gpt2", type=str, help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="gpt2", type=str, help="The model checkpoint for weights initialization.")
    parser.add_argument("--mypretrain_model_name_or_path", default="gpt2", type=str, help="The pretrained infilling model checkpoint for weights initialization.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_generate", action='store_true', help="generate data using trained model")
    parser.add_argument("--continue_train", action='store_true', help="Whether to run training based on a trained checkpoint.")
    parser.add_argument("--no_cuda", action='store_true', help="Avoid using CUDA when available")
    parser.add_argument("--model_parallelize", action='store_true', help="split model on different gpu devices")
    parser.add_argument("--device_count", default=8, type=int, help="total gpu devices")


    parser.add_argument("--sample_in_length", default=100, type=int, help="max source len")
    parser.add_argument("--sample_out_length", default=100, type=int, help="max source len")
    parser.add_argument("--re_length", default=300, type=int, help="max target_len")
    parser.add_argument("--e_length", default=100, type=int, help="max target_len")
    parser.add_argument('--inter_xt_dim', default=512, type=int, help='intermediate dimension size of compressed t from x')
    parser.add_argument('--t_dim', default=256, type=int, help='dimension size of compressed t')
    parser.add_argument("--sample_size", type=int, dest="sample_size", default=5, help='the number of samples to take when estimating the stochastic gradient')
    parser.add_argument('--beta', default=1e-3, type=float, help='beta value on xt loss (tradeoff parameter)')
    parser.add_argument('--gamma', default=1.0, type=float, help='gamma value on x generation')
    parser.add_argument("--uncertainty_loss", action='store_true', help="use uncertainty to get dynamic weights for losses")
    parser.add_argument("--hard_compress_x", action='store_true', help="length of t is 12")
    parser.add_argument('--fix_exp_layers', default=None, type=int, help='fix layers smaller than fix_exp_layers')
    parser.add_argument('--gradient_checkpointing', action='store_true', help="use gradient checkpointing to save memory")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--workers", default=0, type=int, help="workers")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_ratio.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="label_smoothing")
    parser.add_argument("--prefix_dropout", default=0.0, type=float, help="prefix t dropout")


    parser.add_argument("--kl_warmup", default=5, type=float, help="Warmup steps in KL loss for KL annealing")
    parser.add_argument("--kl_beta", default=0.1, type=float, help="Weight of KL loss")
    parser.add_argument('--cycle', type=int, default=4, help='cycle number, cyclical annealing schedule')
    parser.add_argument("--do_ty", action='store_true', help="Whether to generate y given t during training")
    parser.add_argument("--do_tx", action='store_true', help="Whether to generate x given t during training")


    parser.add_argument("--save_last", action='store_true', help="whether save the last epoch")
    parser.add_argument("--evaluate_metrics", default='ppl', type=str, help='choose between ppl and bleu')
    parser.add_argument("--tb_log_dir", default=None, type=str, help="log")
    parser.add_argument('--logging_steps', type=int, default=50, help="Log every X updates steps.")
    parser.add_argument('--validate_steps', type=int, default=2000, help="evaluate model every x updates steps")

    parser.add_argument('--overwrite_output_dir', action='store_true', help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=1024, help="random seed for initialization")
    parser.add_argument("--beam", default=1, type=int,
                        help="beam_size")  # 取出概率最大的k个词构成一个集合，然后将这个子集词的概率再归一化，最后重新的概率分布中采样词汇
    parser.add_argument("--sampling", action='store_true', help="whether topk sampling or topp sampling")
    parser.add_argument("--sampling_topk", default=-1, type=int,
                        help="topk sampling")
    parser.add_argument("--sampling_topp", default=-1, type=float,
                        help="topp sampling")
    parser.add_argument("--temperature", default=1, type=float,
                        help="topp sampling")

    parser.add_argument('--fp16', action='store_true', help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1', help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument("--prediction_dir", default='test', type=str, help='for prediction')
    parser.add_argument("--stop_token", type=str, default='<|endoftext|>', help="Token at which text generation is stopped")

    args = parser.parse_args()
    return args


def get_tokenier(tokenizer_class, load_tokenizer_path, special_tokens=None):
    tokenizer = tokenizer_class.from_pretrained(load_tokenizer_path)
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
        print(str(special_tokens) + "\nSpecial tokens added")
    return tokenizer


def get_model(tokenizer, config_class, model_class, special_tokens=None, load_model_path=None, exp_load_from_path=None, args=None):
    if special_tokens:
        config = config_class.from_pretrained(
            load_model_path,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            unk_token_id=tokenizer.unk_token_id,
            exp_token_id=tokenizer.encode(special_tokens['additional_special_tokens'][0])[0],
            ans_token_id=tokenizer.encode(tokenizer.encode(special_tokens['additional_special_tokens'][1]))[0],
        )
    else:
        config = config_class.from_pretrained(
            load_model_path,
            pad_token_id=tokenizer.eos_token_id)
    if not args.do_train and special_tokens:  ### during eval or continue training
        config.vocab_size = len(tokenizer)

    model = model_class(config=config, args=args, tokenizer=tokenizer)

    if args.do_train:
        model.transformer = model.transformer.from_pretrained(load_model_path, config=config)  ### encoder
        model.transformer.resize_token_embeddings(len(tokenizer))

        if args.model_type == 'gpt2_share_dec':
            model.decoder.transformer = model.decoder.transformer.from_pretrained(load_model_path)
            model.decoder.resize_token_embeddings(len(tokenizer))
        else:
            if args.do_ty:
                model.decoder_for_sample.transformer = model.decoder_for_sample.transformer.from_pretrained(
                    load_model_path)
                model.decoder_for_sample.resize_token_embeddings(len(tokenizer))
            # if args.do_tx:
            model.decoder_for_explanation.transformer = model.decoder_for_explanation.transformer.from_pretrained(exp_load_from_path)
            model.decoder_for_explanation.resize_token_embeddings(len(tokenizer))

        model.config.update({'vocab_size': len(tokenizer)})

        for name, param in model.transformer.named_parameters():  ### we fix the encoder params
            param.requires_grad = False
    else:
        model = model_class.from_pretrained(load_model_path, config=config, args=args, tokenizer=tokenizer)
    return config, model


def put_model_to_device(model, args):
    logger.info("put model to specify gpu devices.")
    if args.model_parallelize:
        device_map_sample = {
            0: [0],
            1: [1, 2, 3, 4, 5, 6],
            2: [7, 8, 9, 10, 11],
            3: [12, 13, 14, 15],
            4: [16, 17, 18, 19],
            5: [20, 21, 22, 23],
            6: [24, 25, 26, 27, 28, 29],
            7: [30, 31, 32, 33, 34, 35],
        }
        device_map_explanation = {
            0: [0],
            1: [1, 2, 3, 4, 5, 6],
            2: [7, 8, 9, 10, 11],
            3: [12, 13, 14, 15],
            4: [16, 17, 18, 19],
            5: [20, 21, 22, 23],
            6: [24, 25, 26, 27, 28, 29],
            7: [30, 31, 32, 33, 34, 35],
        }
        model.parallelize(device_map_sample=device_map_sample, device_map_explanation=device_map_explanation)
    else:
        model.to(args.device)

    return model


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total_num: ', total_num)
    print('trainable_num: ', trainable_num)

    return total_num, trainable_num


class JsonDumpHelper(json.JSONEncoder):
    def default(self, obj):
        if type(obj) != str:
            return str(obj)
        return json.JSONEncoder.default(self, obj)


def set_log(log_file=None, args=None):
    '''
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    :param log_file:
    :return:
    '''
    if args.local_rank in [-1, 0]:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)
    fmt = logging.Formatter('[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                            '%m/%d/%Y %H:%M:%S')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    if log_file != None:
        logfile = logging.FileHandler(log_file, 'w')
        logfile.setFormatter(fmt)
        logger.addHandler(logfile)



def save_generation(args, **kwargs):
    save_result_dir = os.path.join(args.output_dir, "prediction_{}.txt".format(kwargs['prefix']))
    with open(save_result_dir, 'w') as f:
        sample_in = kwargs['sample_in']
        sample_out = kwargs['sample_out']
        expl_ipt = kwargs['expl_ipt_list']
        expl_opt = kwargs['expl_opt_list']
        prediction_x = kwargs['prediction_x']
        vinfo = kwargs['vinfo']
        if 'multi_prediction_x' in kwargs:
            multi_prediction_x = kwargs['multi_prediction_x']
        else:
            multi_prediction_x = None

        for i, line in enumerate(sample_in):
            f.write('-----------' + str(i) + '-----------' + '\n')
            f.write('sample_in: ' + line + '\n')
            f.write('sample_out: ' + sample_out[i] + '\n')
            f.write('expl_ipt: ' + expl_ipt[i] + '\n')
            f.write('expl_opt: ' + expl_opt[i] + '\n')
            f.write('prediction_x: ' + ' '.join(prediction_x[i]) + '\n')
            if multi_prediction_x is not None or args.beam > 1:
                f.write('multi_prediction_x: ' + str(multi_prediction_x[i]) + '\n')
            f.write('vinfo: ' + str(vinfo[i]) + '\n')

    logger.info("Save generation result in {}".format(save_result_dir))


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
    mi_dist2 = len(set(bigrams)) / (float)(len(bigrams))
    avg_len /= n
    return ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len