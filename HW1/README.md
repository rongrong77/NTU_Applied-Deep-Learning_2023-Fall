
## Paragraph Selection
### Train
``` bash
python Paragraph_Selection.py \
--model_name_or_path <model_name> \
--cache_dir "./cache" \
--train_file <train_file> \
--validation_file <valid_file> \
--context_file <context> \
--preprocessing_num_workers 6 \
--output_dir <output_dir> \
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
```
* **model_name:** Pretrained model name.
* **train_file:** Path to tain file.
* **valid_file:** Path to validation file.
* **context_file:** Path to context file.
* **output_dir:** Directory to the output checkpoint.

### Test
``` bash
python Paragraph_Selection.py \
--model_name_or_path= <model_name> \
--cache_dir="./cache" \
--test_file= <test_file> \
--context_file= <test_file> \
--preprocessing_num_workers 6 \
--per_device_eval_batch_size 4 \
--do_predict \
--output_file <output_file>  \
--dataloader_num_workers 6 \
--pad_to_max_length \
--max_seq_length 512 \
```
* **model_name:** Path to pretrained model or model identifier from huggingface.
* **test_file:** Path to test file.
* **context_file:** Path to context file.
* **output_file**: Path to prediction file.

## Span Selection
### Train
``` bash
python Span_Selection.py \
--model_name_or_path <model_name> \
--do_train \
--do_eval \
--train_file <train_file> \
--validation_file <valid_file>  \
--context_file <context_file> \
--output_dir <output_dir> \
``` 
* **model_name:** Path to pretrained model or model identifier from huggingface.
* **train_file:** Path to train file.
* **valid_file:** Path to valid file.
* **context_file:** Path to context file.
* **output_dir:** Directory to the output checkpoint.

### Test
```bash
python Span_Selection.py \
--model_name_or_path <model_name> \
--do_predict \
--test_file <test_file> \
--context_file <context_file> \
--output_file <output_file> \
```
* **model_name:** Path to pretrained model or model identifier from huggingface.
* **test_file:** Path to test file.
* **context_file:** Path to context file.
* **output_file**: Path to prediction file.