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


### Evaluation
``` bash
python ./code/eval.py - ${1} -s ${2}
``` 
# ${1}: path to the input file
# ${2}: path to the output file

# eg: on public file
bash eval.sh ./data/public.jsonl ./sub/submission.jsonl
## result (output of eval.sh)
# eg: on sample test file
{
  "rouge-1": {
    "r": 0.4462301587301588,
    "p": 0.41642011497274656,
    "f": 0.42412794132206677
  },
  "rouge-2": {
    "r": 0.2530990088343029,
    "p": 0.24277884152884152,
    "f": 0.24420569554504906
  },
  "rouge-l": {
    "r": 0.3994444444444444,
    "p": 0.3745911106437423,
    "f": 0.3806626215508284
  }
}


## Train stage

```
bash train.sh

# reinforcement learning
bash train_rl.sh
```

## Test stage

### download

```
bash download.sh
```

### submission

```
bash run.sh /path/to/input.jsonl /path/to/output.jsonl

# eg: on sample test file
bash run.sh ./data/sample_test.jsonl ./tmp/test_output.jsonl

# eg: on public file
bash run.sh ./data/public.jsonl ./tmp/public_output.jsonl
```

### compute metric (tw_rouge)

```
bash eval.sh /path/to/reference.jsonl /path/to/submission.jsonl

# eg: on sample test file
bash eval.sh ./data/sample_submission.jsonl ./tmp/test_output.jsonl

# eg: on public file
bash eval.sh ./data/public.jsonl ./tmp/public_output.jsonl
```

### result (output of eval.sh)

```
# eg: on sample test file
{
  "rouge-1": {
    "r": 0.4462301587301588,
    "p": 0.41642011497274656,
    "f": 0.42412794132206677
  },
  "rouge-2": {
    "r": 0.2530990088343029,
    "p": 0.24277884152884152,
    "f": 0.24420569554504906
  },
  "rouge-l": {
    "r": 0.3994444444444444,
    "p": 0.3745911106437423,
    "f": 0.3806626215508284
  }
}

# eg: on public file
{
  "rouge-1": {
    "r": 0.26525637155156306,
    "p": 0.2845399611414974,
    "f": 0.2667042266450451
  },
  "rouge-2": {
    "r": 0.10771028367466862,
    "p": 0.11330598618332399,
    "f": 0.10699129189320404
  },
  "rouge-l": {
    "r": 0.23680564524843847,
    "p": 0.25436518055402774,
    "f": 0.2380963617304683
  }
}
```
