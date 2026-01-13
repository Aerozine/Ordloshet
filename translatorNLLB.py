import io
import os
import time
import sys
import math
import re
from tqdm import tqdm
from collections import deque
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings

warnings.filterwarnings("ignore")

print("Loading")
model_name = "facebook/nllb-200-3.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("done")


def translate(text, src_lang="eng_Latn", tgt_lang="fra_Latn"):
    if not text or not text.strip():
        return text
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True).to(
        device
    )
    with torch.no_grad():
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=512,
            num_beams=8,
            length_penalty=1.0,
            early_stopping=True
        )
    return tokenizer.decode(translated[0], skip_special_tokens=True)


def largetranslate(text, src_lang="eng_Latn", tgt_lang="fra_Latn"):
    if not text or not text.strip():
        return text
    test_inputs = tokenizer(text, return_tensors="pt")
    input_length = test_inputs["input_ids"].shape[1]
    if input_length <= 400:
        return translate(text, src_lang, tgt_lang)
    # exclamation splitting !?!
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        test_chunk = current_chunk + " " + sentence if current_chunk else sentence
        test_inputs = tokenizer(test_chunk, return_tensors="pt")
        if test_inputs["input_ids"].shape[1] > 400 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk = test_chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    translated_chunks = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk, return_tensors="pt", max_length=512, truncation=True
        ).to(device)
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
                max_length=512,
                num_beams=8,
                length_penalty=1.0,
                early_stopping=True
            )
        translated_chunks.append(
            tokenizer.decode(translated[0], skip_special_tokens=True)
        )
    return " ".join(translated_chunks)
