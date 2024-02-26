## Summarization
### Train
``` bash
python ./code/train.py \
--model_name_or_path <model_name> \
--cache_dir "./cache/" \
--output_dir <output_dir> \
--train_file <train_file> \
--valid_file <valid_file> \
--batch_size 16 \
--lr 1e-4
```
* **model_name:** Pretrained model name. Ex:"google/mt5-small"
* **train_file:** Path to train file. Ex:./data/train.jsonl 
* **valid_file:** Path to validation file. Ex: ./data/public.jsonl
* **output_dir:** Directory to the output checkpoint. Ex:"./best"

### Test
``` bash
python ./code/test.py \
--model_name_or_path <model_name> \
--cache_dir "./cache/" \
--pt_path <pt_file> \
--test_file <test_file> \
--pred_file <pred_file> \
--num_beams 5 \
--top_k 150 \
--top_p 1.0 \
--temperature 1.0
```
* **model_name:** Pretrained model name. Ex:"google/mt5-small"
* **pt_path:** Path to the best checkpoint model generated during training. Ex:./best/best_model.pt
* **test_file:** Path to test file. Ex: ./data/public.jsonl
* **pred_file**: Path to predict file. Ex: ./sub/submission.jsonl


### Evaluation/compute metric (tw_rouge)
``` bash
python ./code/eval.py - ${1} -s ${2}
``` 
# ${1}: path to the input file
# ${2}: path to the output file

# eg: on public test file
bash eval.sh ./data/public.jsonl ./data/submission.jsonl
## result 
{
  "rouge-1": {
    "r": 0.24955753803015523,
    "p": 0.2828006647400814,
    "f": 0.257451235794295
  },
  "rouge-2": {
    "r": 0.10017031269546886,
    "p": 0.11226061467109853,
    "f": 0.1024625533467368
  },
  "rouge-l": {
    "r": 0.22328038390887067,
    "p": 0.2532930595566262,
    "f": 0.2302855181568176
  }
}

## Reproduce my result

bash download.sh
bash run.sh /path/to/input.jsonl /path/to/output.jsonl
bash eval.sh /path/to/reference.jsonl /path/to/submission.jsonl