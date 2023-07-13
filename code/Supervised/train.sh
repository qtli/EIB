TASK_NAME=$1
DATA_PATH=data/
SAVE_PATH=${TASK_NAME}
PRETRAIN_PATH=$2
CUDA_VISIBLE_DEVICES=$3
export CUDA_VISIBLE_DEVICES

python -m torch.distributed.launch --nproc_per_node=8 train.py \
--data_file ${DATA_PATH} \
--data_type ecqa \
--train_data_file ecqa_train.csv \
--dev_data_file ecqa_dev.csv \
--test_data_file ecqa_test.csv \
--output_dir ${SAVE_PATH} \
--task_type ${TASK_NAME} \
--source_length 200 \
--target_length 50 \
--model_type gpt2 \
--model_name_or_path ${PRETRAIN_PATH} \
--do_train \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics ppl \
--validate_steps -1 \
--overwrite_output_dir \
--num_train_epochs 20 \
--learning_rate 5e-5 \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--logging_steps 0