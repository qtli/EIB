#!/bin/bash
PROMPT_DATA=$1
FILTER_RESULT=$2
TYPE=$3
python get_prompt_filter_result.py --prompt_data_file ${PROMPT_DATA} --filter_result ${FILTER_RESULT} --type {TYPE}
