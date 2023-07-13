#!/bin/bash
SOURCE_FILE=$1
NEXT_FILE=$2
OUTPUT_FILE=$3

CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES

python bottleEx_summarize.py -S1_path ${SOURCE_FILE} -S2_path ${NEXT_FILE} -rem_words 3 -start_idx 0 -end_idx 999999 -out_name ${OUTPUT_FILE}