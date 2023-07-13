#!/usr/bin/env python
#coding:utf-8
import pdb
import os
import sys
import torch.multiprocessing as mp
# os.environ('LOCAL_RANK')
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv('CUDA_VISIBLE_DEVICES')  # must before 'import torch'
import subprocess
import warnings
warnings.filterwarnings('ignore')
from collections import Counter
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

import json
from contextlib import contextmanager
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import GPT2Model, GPT2Tokenizer
from transformers import set_seed, CONFIG_NAME, WEIGHTS_NAME
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM

import logging
logger = logging.getLogger()

proj_dir = '/'.join(os.getcwd().split('/')[:-2])
sys.path.insert(0, proj_dir)
sys.path.insert(0, os.path.dirname(__file__))
set_seed(1024)

from dataset import DatasetHelper, DatasetHelper_esnli
from util import _compute_bleu, _get_dist, set_log, JsonDumpHelper, save_generation, setup_args, get_tokenier, get_model
from optimization import AdamW, WarmupLinearSchedule, BalancedDataParallel
from modeling_gpt2 import TextGen, GPT2Config
import wandb

MODEL_CLASSES = {
    'gpt2': (GPT2Config, TextGen, GPT2Tokenizer),
}

SPECIAL_TOKENS = {"bos_token": "<|bos|>",
                  "eos_token": "<|endoftext|>",
                  "unk_token": "<|u|>",
                  "pad_token": "<|pad|>",
                  "additional_special_tokens": ["<|exp|>", "<|ans|>"],
                  }

args = setup_args()

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    '''
    decorator to make all processes in distributed training wait for each local master to do something.
    :param local_rank:
    :return:
    '''
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()  ### todo

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        drop_last=True,
        num_workers=args.workers,
        shuffle=False,
        pin_memory=True,  # pin_memory=True when loading data on CPU to GPU
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    base_params = []
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(gn in n for gn in base_params)]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=int(args.warmup_ratio * t_total), t_total=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # model.set_checkpoint(args.use_checkpoint)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        # model = BalancedDataParallel(4, model, dim=0)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    if args.local_rank in [-1,0]:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size * args.gradient_accumulation_steps * (
                        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)


    global_step = 0
    patient = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_valid = {'bleu': 0.0, 'ppl': np.Inf, 'acc': 0.0}
    if args.validate_steps == -1: args.validate_steps = len(train_dataloader)

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1,0])

    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        local_step = 0

        for step, batch in enumerate(epoch_iterator):
            batch = {key: batch[key].to(args.device) for key in batch}
            model.train()
            all_loss = model(state='train', **batch)
            loss = all_loss['train/tot_loss'].mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()   ### communication between different gpus

            tr_loss += loss.item()
            epoch_iterator.set_postfix(tot=loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                local_step += 1

            if args.local_rank in [-1, 0] and global_step % args.validate_steps == 0:
                sign_list = {'ppl': 1.0, 'bleu': -1.0, 'acc': -1.0}
                result = evaluate(args, model, tokenizer, args.evaluate_metrics, prefix=epoch)
                logger.info('step: {}'.format(step))
                logger.info("Epoch {} evaluate {}: {:.4f}".format(epoch, args.evaluate_metrics, result[args.evaluate_metrics]))

                if args.use_wandb:
                    wandb_eval_result = result.copy()
                    for key in list(wandb_eval_result.keys()):
                        wandb_eval_result["eval/{}".format(key)] = wandb_eval_result.pop(key)
                    wandb.log(wandb_eval_result)

                if (result[args.evaluate_metrics] - best_valid[args.evaluate_metrics]) * sign_list[args.evaluate_metrics] < 0:
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(args.output_dir)  # WEIGHTS_NAME
                    # torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, WEIGHTS_NAME))
                    tokenizer.save_pretrained(args.output_dir)
                    model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
                    torch.save(args, os.path.join(args.output_dir, 'training_args.bin')) # Good practice: save your training arguments together with the trained model
                    torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.bin"))
                    subprocess.call(["cp", os.path.join(args.model_name_or_path, "vocab.json"), args.output_dir])
                    subprocess.call(["cp", os.path.join(args.model_name_or_path, "merges.txt"), args.output_dir])

                    best_valid[args.evaluate_metrics] = result[args.evaluate_metrics]
                    patient = 0
                    logger.info("Saving model checkpoint to %s", args.output_dir)
                    logger.info('Patient is {}.'.format(patient))
                else:
                    patient += 1
                    logger.info('Patient is {}.'.format(patient))
                    if patient > 2:
                        logger.info('Patient is {} and Stop training.'.format(patient))
                        break
            if patient > 2:
                break
        if patient > 2:
            break

    with torch_distributed_zero_first(args.local_rank):
        if args.save_last:  #  and (args.local_rank in [-1,0] or torch.distributed.get_rank() == 0)
            model_to_save = model.module if hasattr(model,'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            model_to_save.config.to_json_file(os.path.join(args.output_dir, CONFIG_NAME))
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'last_training_args.bin'))
            torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "last_scheduler.bin"))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "last_optimizer.bin"))
            subprocess.call(["cp", os.path.join(args.model_name_or_path, "vocab.json"), args.output_dir])
            subprocess.call(["cp", os.path.join(args.model_name_or_path, "merges.txt"), args.output_dir])
            logger.info("Saving model checkpoint to %s", args.output_dir)

        tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate(args, model, tokenizer, evaluate_metrics="ppl", prefix='0'):
    eval_output_dir = args.output_dir
    if prefix == 'test':
        eval_data_file = os.path.join(args.data_file, args.test_data_file)
    elif prefix == 'train':
        eval_data_file = os.path.join(args.data_file, args.train_data_file)
    else:
        prefix = 'dev'
        eval_data_file = os.path.join(args.data_file, args.dev_data_file)

    if args.data_type == 'ecqa':
        dh = DatasetHelper
    elif args.data_type == 'esnli':
        dh = DatasetHelper_esnli
    else:
        dh = None
    eval_dataset = dh(args,
                      tokenizer,
                      special_tokens=SPECIAL_TOKENS,
                      data_path=eval_data_file,
                      max_length=args.source_length,
                      do_generate=('bleu' in evaluate_metrics))
    eval_dataset.load(
        max_size=None,
        total_batch_size=args.per_gpu_eval_batch_size * max(1, args.n_gpu),
        gpu_num=torch.distributed.get_world_size() if args.local_rank != -1 else 1,
    )
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]: os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank in [-1,0] else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        drop_last=False if args.do_eval else True,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False
    )
    if getattr(eval_dataset, "print_features", False):
        eval_dataset.print_features()

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    gen_seqs = []
    input_list, target_list = [],[]
    nb_eval_steps = 0
    model.eval()

    Hit_num = 0
    for idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
        if args.do_eval:
            print('input: ', batch[1])
            print('target: ', batch[2])

            input_list.extend(batch[1])
            target_list.extend(batch[2])
            batch = batch[0]
            batch = {key: batch[key].to(args.device) for key in batch}

        with torch.no_grad():
            if 'bleu' in evaluate_metrics or 'dist' in evaluate_metrics:
                if isinstance(model, torch.nn.DataParallel):
                    hypo_results = model.module.aotoreg_generate(args=args, **batch)
                else:
                    hypo_results = model.aotoreg_generate(args=args, **batch)
                gen_seqs.append(hypo_results)
                print('hypos: ', hypo_results)

            if 'ppl' in evaluate_metrics:
                eval_loss += model(state='eval', **batch)['eval/tot_loss'].mean().item()
        nb_eval_steps += 1
    result = {}

    if 'ppl' in evaluate_metrics:
        eval_loss = eval_loss / nb_eval_steps
        perplexity = torch.exp(torch.tensor(eval_loss))
        result["ppl"] = perplexity

    if 'bleu' in evaluate_metrics or 'dist' in evaluate_metrics:
        references = [[[x for x in p.strip(' ').split(' ')]] for p in target_list]  # each sample can have multiple references
        predictions = [xs[0].strip(' ').split(' ') for xs in gen_seqs]

        print_data = {'prefix': prefix,
                      'input': input_list,
                      'target': target_list,
                      'prediction': predictions,
                      }
        save_generation(args, **print_data)

        if 'bleu' in evaluate_metrics:
            ipt = Counter({'reference_corpus': references, 'translation_corpus': predictions})
            bleu1 = _compute_bleu(**ipt, max_order=1)
            bleu2 = _compute_bleu(**ipt, max_order=2)
            bleu4 = _compute_bleu(**ipt, max_order=4)
            if prefix == 'test':
                result["blue1"] = bleu1[0]
                result["blue2"] = bleu2[0]
                result["bleu4"] = bleu4[0]
            else:
                result["bleu"] = bleu4[0]

        if 'dist' in evaluate_metrics:
            ma_dist1, ma_dist2, mi_dist1, mi_dist2, avg_len = _get_dist(predictions)
            result['dist-1'] = mi_dist1 * 100
            result['dist-2'] = mi_dist2 * 100

    if 'acc' in evaluate_metrics:
        result = {
            "acc": Hit_num / len(eval_dataset),
        }
    logger.info(result)
    return result


def main():
    if args.use_wandb and args.local_rank in [-1,0]:
        wandb.login(key='8185e8129520afa4aa362d19152d719396262bec')
        wandb.init(project="Explanation", entity="qintongli", name=args.task_type)

    ##### load start checkpoint
    if args.do_eval:
        load_from_path = args.output_dir
        if args.beam > 1:
            args.prediction_dir = '_beam' + str(args.beam) + \
                                  '_topk' + str(args.sampling_topk) + \
                                  '_temp' + str(args.temperature) + \
                                  '_' + args.prediction_dir
        args.output_dir = os.path.join(args.output_dir, args.prediction_dir)
    elif args.continue_train:
        load_from_path = args.output_dir
        if args.local_rank in [-1,0]: logger.info('Continue training from {}'.format(args.output_dir))
    else:
        load_from_path = args.model_name_or_path

    ##### Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1,0]: os.makedirs(args.output_dir)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    ##### Setup CUDA, GPU & distributed training
    # 首先，为了区分不同的进程，初始时需要对每个进程传入rank。这里一般分为node_rank和local_rank，分别表示进程处于哪一机器以及同机器上处于第几进程。如果在单一机器上，node_rank可以省略。local_rank一般通过命令行参数 - -local_rank传入，为int类型。也可以通过环境变量传入local_rank，只需在torch.distributed.launch时，使用 - -use_env参数。无论哪种方式，在训练脚本中，都要获取到local_rank，用于初始化分布式通信，以及区分进程。如果你使用fastNLP，可以通过fastNLP.get_local_rank来得到local_rank。
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    ##### Setup logging
    # set_log(os.path.join(args.output_dir, "log.txt"), args=args)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    ##### Setup seed
    set_seed(args.seed)
    if args.local_rank in [-1, 0]:
        logger.info('-' * 100)
        logger.info('CONFIG:\n%s' % json.dumps(vars(args), cls=JsonDumpHelper, indent=4, sort_keys=True))
        logger.info('-' * 100)
        logger.info('loading model checkpoint from {}'.format(load_from_path))

    with torch_distributed_zero_first(args.local_rank):
        ###  Barrier to make sure only the first process in distributed training download model & vocab & process the dataset. and the others will use the cache
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        tokenizer = get_tokenier(tokenizer_class=tokenizer_class, special_tokens=SPECIAL_TOKENS, load_tokenizer_path=load_from_path)
        config, model = get_model(tokenizer=tokenizer, config_class=config_class, model_class=model_class, special_tokens=SPECIAL_TOKENS, load_model_path=load_from_path, args=args)

    model.to(args.device)
    logger.info('current tokenizer size is {}'.format(len(tokenizer)))
    logger.info('Model lm_head: ' + str(model.transformer.lm_head))

    hyper_params = vars(args).copy()
    for key in list(hyper_params.keys()):
        hyper_params["hyper_p_{}".format(key)] = hyper_params.pop(key)
    base_config = vars(config).copy()
    for key in list(base_config.keys()):
        base_config["{}_{}".format(args.model_type, key)] = base_config.pop(key)

    if args.use_wandb and args.local_rank in [-1,0]:
        wandb.config.update(hyper_params)
        wandb.config.update(base_config)

    if args.do_train or args.continue_train:
        if args.data_type == 'ecqa':
            dh = DatasetHelper
        elif args.data_type == 'esnli':
            dh = DatasetHelper_esnli
        else:
            dh = None
        with torch_distributed_zero_first(args.local_rank):
            train_dataset = dh(
                args,
                tokenizer,
                special_tokens=SPECIAL_TOKENS,
                data_path=os.path.join(args.data_file, args.train_data_file),
                max_length=args.source_length)
            train_dataset.load(
                max_size=None,
                total_batch_size=args.per_gpu_train_batch_size * max(1, args.n_gpu) * args.gradient_accumulation_steps,  #  * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
                gpu_num=torch.distributed.get_world_size() if args.local_rank != -1 else 1,
            )
            if getattr(train_dataset, "print_features", False):
                train_dataset.print_features()
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        with torch_distributed_zero_first(args.local_rank):
            logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if args.do_eval:
        with torch_distributed_zero_first(args.local_rank):
            prefix = 'test'
            result = evaluate(args, model, tokenizer, args.evaluate_metrics, prefix)
            logger.info("Test evaluate {}".format(args.evaluate_metrics))
            metric_result_dir = os.path.join(args.output_dir, "metric_{}.txt".format(prefix))
            args_result_dir = os.path.join(args.output_dir, "args_{}.json".format(prefix))
            with open(metric_result_dir, 'w') as f:
                for k in result:
                    logger.info("{}: {:.4f}".format(k, result[k]))
                    f.write("{}: {:.4f}\n".format(k, result[k]))
            with open(args_result_dir, 'w') as f:
                json.dump(vars(args), f, cls=JsonDumpHelper, indent=4, sort_keys=True)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()