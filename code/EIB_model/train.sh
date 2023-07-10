#!/bin/bash
TASK_NAME=$1
DATA_PATH=$2
SAVE_PATH=${TASK_NAME}_train_cpt/
PRETRAIN_PATH=$3
EXP_PRETRAIN_PATH=$3

CUDA_VISIBLE_DEVICES=$4
export CUDA_VISIBLE_DEVICES

python -m torch.distributed.prediction_dirlaunch --nproc_per_node=8 train.py \
--data_file ${DATA_PATH} \
--data_type unify \
--train_data_file train.csv \
--dev_data_file dev.csv \
--output_dir ${SAVE_PATH} \
--task_type ${TASK_NAME} \
--sample_in_length 150 \
--re_length 300 \
--e_length 300 \
--inter_xt_dim 512 \
--t_dim 256 \
--hard_compress_x \
--sample_size 5 \
--beta 0.0001 \
--gamma 1.0 \
--kl_warmup 0 \
--kl_beta 1.0 \
--cycle 4 \
--do_tx \
--do_ty \
--model_type gpt2 \
--model_name_or_path ${PRETRAIN_PATH} \
--mypretrain_model_name_or_path ${EXP_PRETRAIN_PATH} \
--do_train \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 1 \
--workers 0 \
--seed 42 \
--evaluate_metrics ppl \
--validate_steps -1 \
--overwrite_output_dir \
--num_train_epochs 20 \
--learning_rate 5e-5 \
--logging_steps 0