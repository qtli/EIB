#!/bin/bash
TASK_NAME=$1
DATA_PATH=$2
SAVE_PATH=$3
PRETRAIN_PATH=$4
EXP_PRETRAIN_PATH=$4
if [[ "$DATA_PATH" == *"ecqa"* ]]
then
  DATA_TYPE=ecqa
  TEST_DATA=prompt_ecqa_test.csv
elif [[ "$DATA_PATH" == *"esnli"* ]]
then
  DATA_TYPE=esnli
  TEST_DATA=prompt_esnli_test.csv
else
  DATA_TYPE=mixexpl_test
  TEST_DATA=test.csv
fi

CUDA_VISIBLE_DEVICES=$5
export CUDA_VISIBLE_DEVICES

python -m torch.distributed.launch --nproc_per_node=1 --master_port 29502 train.py \
--data_file ${DATA_PATH} \
--data_type ${DATA_TYPE} \
--test_data_file ${TEST_DATA} \
--output_dir ${SAVE_PATH} \
--prediction_dir prediction_${DATA_TYPE} \
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
--do_eval \
--per_gpu_train_batch_size 4 \
--per_gpu_eval_batch_size 1 \
--workers 0 \
--seed 42 \
--evaluate_metrics ppl_bleu_dist \
--validate_steps -1 \
--overwrite_output_dir \
--num_train_epochs 20 \
--learning_rate 5e-5 \
--logging_steps 0
