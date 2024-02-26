#!/bin/bash

#${1}: path to the folder containing the files downloaded by download.sh
#${2}: path to the input file (.json)
#${3}: path to the output file (.json)

python .code/inference.py \
    --base_model_path "yentinglin/Taiwan-LLM-7B-v2.0-chat" \
    --peft_path ${1} \
    --test_data_path ${2} \
    --output_path ${3}

