python Paragraph_Selection.py \
--model_name_or_path="./PS_rol/" \
--cache_dir="./cache" \
--test_file="${2}" \
--context_file="${1}" \
--preprocessing_num_workers=6 \
--per_device_eval_batch_size=4 \
--output_dir="./" \
--do_predict \
--output_file="./test_for_SS.json"  \
--dataloader_num_workers=6

python Span_Selection.py \
--model_name_or_path="./SS_roberta_large/" \
--do_predict \
--test_file="./test_for_SS.json" \
--context_file="${1}" \
--output_file="${3}"