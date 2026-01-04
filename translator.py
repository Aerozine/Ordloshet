import io 
import os
import time
import sys
import math 
from tqdm  import tqdm
from collections import deque
import torch
from transformers import AutoProcessor
from transformers import SeamlessM4Tv2Model
# Need to fix hardcoding
print("load processor")
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
print("load model")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
#model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
#model = SeamlessM4Tv2ForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
print("load device")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print("done")
def translate(text,src_lang="eng",tgt_lang="fra"):
    text_inputs=processor(text=text,src_lang=src_lang,return_tensors="pt").to(device)
    output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
    output = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    return output
