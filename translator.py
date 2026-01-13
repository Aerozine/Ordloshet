import io
import os
import time
import sys
import math
from tqdm import tqdm
from collections import deque
import torch
from transformers import AutoProcessor
from transformers import SeamlessM4Tv2Model, SeamlessM4Tv2ForTextToText
import torch
import warnings

print("loading")
warnings.filterwarnings("ignore", message=".*layer_idx.*")
""" Seamless m4T """
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print("done")


def translate(text, src_lang="eng", tgt_lang="fra"):
    text_inputs = processor(text=text, src_lang=src_lang, return_tensors="pt").to(
        device
    )
    output_tokens = model.generate(
        **text_inputs, tgt_lang=tgt_lang, generate_speech=False
    )
    output = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return output


def largetranslate(text, src_lang="eng", tgt_lang="fra"):
    if not text or not text.strip():
        return text
    test_inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
    input_length = test_inputs["input_ids"].shape[1]
    if input_length <= 450:
        text_inputs = test_inputs.to(device)
        with torch.no_grad():
            output_tokens = model.generate(
                **text_inputs, tgt_lang=tgt_lang, num_beams=5, max_new_tokens=512
            )
        return processor.decode(output_tokens[0], skip_special_tokens=True)
    # exclamation splitting !?!
    sentences = re.split(r"([.!?]+\s+)", text)
    chunks = []
    current_chunk = ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        punct = sentences[i + 1] if i + 1 < len(sentences) else ""
        segment = sentence + punct
        test_chunk = current_chunk + segment
        test_inputs = processor(text=test_chunk, src_lang=src_lang, return_tensors="pt")
        if test_inputs["input_ids"].shape[1] > 450 and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = segment
        else:
            current_chunk += segment
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    translated_chunks = []
    for chunk in chunks:
        text_inputs = processor(text=chunk, src_lang=src_lang, return_tensors="pt").to(
            device
        )
        with torch.no_grad():
            output_tokens = model.generate(
                **text_inputs, tgt_lang=tgt_lang, num_beams=5, max_new_tokens=512
            )
        translated_chunks.append(
            processor.decode(output_tokens[0], skip_special_tokens=True)
        )
    return " ".join(translated_chunks)
