import os
import re
import gc

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import LoraConfig, get_peft_model, TaskType
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

import torch
import torch.nn as nn

if torch.cuda.is_available():       
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

os.environ["HF_HOME"] = "E:\Huggingface"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained('./fine-tuned-llama2-7b-chat-hf-aistudio-finance-QLoRA3-subset')
llm = AutoModelForCausalLM.from_pretrained(
    './fine-tuned-llama2-7b-chat-hf-aistudio-finance-QLoRA3-subset',
    # cache_dir="./model_cache/models--meta-llama--Llama-2-7b-chat-hf/",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config, 
    device_map=device 
)
model_pipeline = pipeline("text-generation", model=llm, tokenizer=tokenizer, max_new_tokens=1024)