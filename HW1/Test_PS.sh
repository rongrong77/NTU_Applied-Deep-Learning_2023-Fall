python Paragraph_Selection.py \
--model_name_or_path="./PS_rol/" \
--cache_dir="./cache" \
--test_file="./data/test.json" \
--context_file="./data/context.json" \
--preprocessing_num_workers=6 \
--per_device_eval_batch_size=4 \
--output_dir="./" \
--do_predict \
--output_file="./test_for_QA_rol.json"  \
--dataloader_num_workers=6 \
--pad_to_max_length \
--max_seq_length 512 \