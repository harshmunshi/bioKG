#!/usr/bin/env python3
"""
Interactive prediction loop against the naive (vanilla, pretrained) base LLM —
no KG embeddings, no projection layer, no LoRA adapters, no entity linking.

Useful as a quick baseline to compare against BioKG-LoRA's grounded answers,
without going through train.py's full config/pipeline machinery.

Usage:
    python scripts/predict_baseline.py
    python scripts/predict_baseline.py --base-model meta-llama/Llama-3-8B
    python scripts/predict_baseline.py --question "What phenotypes does Thbd knockout cause?"
    python scripts/predict_baseline.py --quantization 8bit --no-sample
"""
from __future__ import annotations

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Run predictions on the naive base LLM (no BioKG-LoRA).")
    parser.add_argument("--base-model", type=str, default="google/gemma-4-e2b-it",
                        help="HuggingFace model id (default: google/gemma-4-e2b-it)")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"],
                        help="Quantization mode (default: 4bit)")
    parser.add_argument("--question", type=str, default=None,
                        help="Ask a single question and exit instead of looping")
    parser.add_argument("--use-template", action="store_true",
                        help="Wrap the question in BioKG-LoRA's QA prompt template "
                             "(### Biological Reasoning Question / ### Answer) instead of asking it raw")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--no-sample", action="store_true", help="Greedy decoding instead of sampling")
    return parser.parse_args()


def load_model(base_model: str, quantization: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    print(f"Loading {base_model} (quantization={quantization})...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if quantization == "none" else None,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


@torch.no_grad()
def answer(model, tokenizer, question: str, args) -> str:
    if args.use_template:
        prompt = f"### Biological Reasoning Question\n{question}\n\n### Answer\n"
    else:
        prompt = question

    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)

    out_ids = model.generate(
        input_ids=enc["input_ids"],
        attention_mask=enc["attention_mask"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    args = parse_args()
    model, tokenizer = load_model(args.base_model, args.quantization)

    if args.question:
        result = answer(model, tokenizer, args.question, args)
        print(f"\n{'=' * 60}\nQuestion: {args.question}\n{'-' * 60}\nAnswer:\n{result}\n{'=' * 60}")
        return

    print("\nNaive base-model prediction loop. Type a question, or 'quit'/'exit' to stop.\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            break
        result = answer(model, tokenizer, question, args)
        print(f"\n{'-' * 60}\nAnswer:\n{result}\n{'-' * 60}\n")


if __name__ == "__main__":
    main()
