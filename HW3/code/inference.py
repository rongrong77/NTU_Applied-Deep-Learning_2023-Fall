import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from peft import PeftModel
from utils import get_prompt, get_bnb_config
import argparse

def generate_predictions(model, tokenizer, data, max_length=512, batch_size=4):
    data_size = len(data)
    predictions = []

    for i in tqdm(range(0, data_size, batch_size), desc="Generating Predictions"):
        batch_data = data[i:i + batch_size]
        batch_instructions = [get_prompt(item["instruction"]) for item in batch_data]

        input_ids = tokenizer(batch_instructions, return_tensors='pt', padding=True, truncation=True)['input_ids'].to("cuda")

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = model.generate(input_ids=input_ids, max_length=max_length).to("cpu")

        for j, example in enumerate(batch_data):
            generated_text_ids = outputs[j][outputs[j] != tokenizer.pad_token_id]
            generated_text = tokenizer.decode(generated_text_ids, skip_special_tokens=True)

            # Split the generated_text using "ASSISTANT:" as a delimiter
            _, assistant_part = generated_text.split("ASSISTANT:", 1)
            assistant_part = assistant_part.strip()

            predictions.append({'id': example['id'], 'output': assistant_part})
        model.to("cuda")
        torch.cuda.empty_cache()
    return predictions



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="predictions.json",
        help="Path to save the generated predictions."
    )
    args = parser.parse_args()


    if args.base_model_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16,
            quantization_config=get_bnb_config()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model_path,
            padding_side="left",
            trust_remote_code=True
        )
    else:
        model_name = "yentinglin/Taiwan-LLM-7B-v2.0-chat"
        revision = "5073b2bbc1aa5519acdc865e99832857ef47f7c9"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            quantization_config=get_bnb_config()
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            padding_side="left",
            trust_remote_code=True
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load LoRA
    model = PeftModel.from_pretrained(model, args.peft_path)

    with open(args.test_data_path, "r",encoding="utf-8") as f:
        data = json.load(f)

    model.eval()

    predictions = generate_predictions(model, tokenizer, data)
    
    # Save predictions to a JSON file
    with open(args.output_path, "w",encoding="utf-8") as output_file:
        json.dump(predictions, output_file, indent=2,ensure_ascii=False)

