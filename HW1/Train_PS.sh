#!/bin/bash -x

python Paragraph_Selection.py \
--model_name_or_path "hfl/chinese-roberta-wwm-ext" \
--cache_dir "./cache" \
--train_file "./data/train.json" \
--validation_file "./data/valid.json" \
--context_file "./data/context.json" \
--preprocessing_num_workers 6 \
--output_dir "./PP_rol/" \
--pad_to_max_length \
--max_seq_length 512 \
--do_train \
--do_eval \
--num_train_epochs 1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 3e-5 \
--warmup_steps 300 \
--dataloader_num_workers 6 \
--evaluation_strategy "steps" \
--eval_steps 500 \
--save_steps 500 \
--metric_for_best_model "accuracy" \
--load_best_model_at_end \
--report_to "tensorboard" \
--fp16 \
--overwrite_output_dir \
