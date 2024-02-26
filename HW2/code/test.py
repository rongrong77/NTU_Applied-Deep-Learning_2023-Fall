from argparse import ArgumentParser, Namespace

import jsonlines
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
)

class MT5Dataset(Dataset):

    def __init__(self, dataset, tokenizer, mode, max_input=512, max_output=64):
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.mode = mode

        self.max_input = max_input
        self.max_output = max_output
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        c_dataset = self.dataset[index]

        text = self.tokenizer(
            [c_dataset["text"]],
            max_length=self.max_input,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        text_seq = text["input_ids"].squeeze()
        text_mask = text["attention_mask"].squeeze()

        summary_seq = summary_mask = []
        if self.mode != 'test':
            with self.tokenizer.as_target_tokenizer():
                # label
                summary = self.tokenizer(
                    [c_dataset["summary"]],
                    max_length=self.max_output,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt",
                )
            summary_seq = summary["input_ids"].squeeze()
            summary_mask = summary["attention_mask"].squeeze()

        c_item = {
            "id": self.dataset[index]['id'],
            "text_seq": text_seq,
            "text_mask": text_mask,
            "summary_seq": summary_seq,
            "summary_mask": summary_mask,
        }
        return c_item

def preprocess_data(input_file):
    _list = []

    # Read the input JSONL file
    with jsonlines.open(input_file) as f_in:
        for obj in f_in:
            _d = {
                "id": obj["id"],
                "text": "" if "maintext" not in obj else obj["maintext"],
                "summary": "" if "title" not in obj else obj["title"],
            }
            _list.append(_d)
    return _list

def main(args):
    print("*** from_pretrained ***")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir,
                                              use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir)
    model.to(args.device)

    print("*** Dataset ***")
    test_data = preprocess_data(args.test_file)
    test_dataset = MT5Dataset(dataset=test_data,
                               tokenizer=tokenizer,
                               mode='test',
                               max_input=args.max_input,
                               max_output=args.max_output)

    print("*** DataLoader ***")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    print("*** model load ***")
    model.load_state_dict(torch.load(args.pt_path))
    model.eval()

    print("*** Testing ***")
    with torch.no_grad():
        ids, pred_seqs = [], []
        for d in tqdm(test_loader):
            text_seq, text_mask = d['text_seq'].to(
                args.device), d['text_mask'].to(args.device)
            id = d['id']

            pred_seq = model.generate(
                input_ids=text_seq,
                attention_mask=text_mask,
                max_length=args.max_output,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )

            for i, p in zip(id, pred_seq):
                _p = tokenizer.decode(p, skip_special_tokens=True)
                ids.append(i)
                pred_seqs.append(_p)

    with jsonlines.open(args.pred_file, mode="w") as f:
        for i, p in zip(ids, pred_seqs):
            f.write({"title": p, "id": i})

    return


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--pt_path", type=str)

    parser.add_argument("--test_file", type=str)
    parser.add_argument("--pred_file", type=str)

    parser.add_argument("--max_input", type=int, default=256)
    parser.add_argument("--max_output", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--num_beams", type=int, default=5)  
    parser.add_argument("--do_sample", action="store_true", default=False) 
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
