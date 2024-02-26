from argparse import ArgumentParser, Namespace

import os, json, random
import jsonlines
import numpy as np
from tqdm import trange, tqdm

import torch
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
)
from torch.utils.data import Dataset
from tw_rouge import get_rouge

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
    best_score = 0.0  # Initialize with a low value
    best_epoch = 0
    print("*** from_pretrained ***")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir,
                                              use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir)
    model.to(args.device)

    print("*** Dataset ***")
    train_data = preprocess_data(args.train_file)
    validation_data = preprocess_data(args.valid_file)

    train_dataset = MT5Dataset(dataset=train_data,
                               tokenizer=tokenizer,
                               mode='train',
                               max_input=args.max_input,
                               max_output=args.max_output)
    eval_dataset = MT5Dataset(dataset=validation_data,
                              tokenizer=tokenizer,
                              mode='validation',
                              max_input=args.max_input,
                              max_output=args.max_output)
    
    print("*** DataLoader ***")
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    eval_loader = DataLoader(dataset=eval_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  weight_decay=args.weight_decay)

    print("*** Training & Validation ***")
    results = {}
    for epoch in trange(1, args.num_epoch+1):
        model.train()
        train_loss = []
        for i, d in enumerate(tqdm(train_loader)):
            text_seq, text_mask = d['text_seq'].to(
                args.device), d['text_mask'].to(args.device)
            summary_seq, summary_mask = d['summary_seq'].to(
                args.device), d['summary_mask'].to(args.device)

            loss = model(input_ids=text_seq,
                         labels=summary_seq,
                         attention_mask=text_mask,
                         decoder_attention_mask=summary_mask).loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        print(f"\nTraining loss: {train_loss[-1] / len(train_loader)}\n",
              flush=True)
        #torch.save(model.state_dict(), f"{args.output_dir}/{epoch}.pt")

        model.eval()
        with torch.no_grad():
            pred_seqs, summary_seqs = [], []
            for d in tqdm(eval_loader):
                text_seq, text_mask = d['text_seq'].to(
                    args.device), d['text_mask'].to(args.device)
                summary_seq, summary_mask = d['summary_seq'].to(
                    args.device), d['summary_mask'].to(args.device)

                pred_seq = model.generate(input_ids=text_seq,
                                          attention_mask=text_mask,
                                          max_length=args.max_output,
                                          num_beams=2)
                for p, l in zip(pred_seq, summary_seq):
                    _p = tokenizer.decode(p, skip_special_tokens=True)
                    _l = tokenizer.decode(l, skip_special_tokens=True)
                    if _p:
                        pred_seqs.append(_p)
                        summary_seqs.append(_l)

            result = get_rouge(pred_seqs, summary_seqs)
            for k in result.keys():
                result[k]['f'] *= 100
            print(f"\nValidation get_rouge: {result}\n", flush=True)

        results[epoch] = result
        eval_score =(results[epoch]['rouge-1']['f']/22.0+
                    results[epoch]['rouge-2']['f']/8.5+
                    results[epoch]['rouge-l']['f']/20.5)

    with open(f"{args.output_dir}/eval_result.json", "w") as f:
        json.dump(results, f, indent=4)
    if eval_score > best_score:
        best_score = eval_score
        best_epoch = epoch
        torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")

    print(f"Best model checkpoint is from epoch {best_epoch}")

    return


def set_seed(seed):
    ## insure the reproducibility ##
    # Ref. https://pytorch.org/docs/stable/notes/randomness.html

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=5487)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--output_dir", type=str)

    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)

    parser.add_argument("--max_input", type=int, default=256)
    parser.add_argument("--max_output", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--num_epoch", type=int, default=25)  # 10, 15, 20, 25

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
