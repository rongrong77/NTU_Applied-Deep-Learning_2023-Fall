#!/bin/bash

# ${1}: path to the input file
# ${2}: path to the output file

python ./code/test.py \
    --model_name_or_path "./mt5-small" \
    --cache_dir "./best/cache/" \
    --pt_path "./best/best_model.pt" \
    --test_file ${1} \
    --pred_file ${2}
