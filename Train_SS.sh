python Span_Selection.py \
--model_name_or_path "hfl/chinese-roberta-wwm-ext-large" \
--do_train \
--do_eval \
--train_file "data/train.json" \
--validation_file "data/valid.json" \
--context_file "data/context.json" \
--output_dir "SS_rol" \