# pip install -q -U git+https://github.com/huggingface/transformers.git
# pip install -q -U git+https://github.com/huggingface/peft.git
# pip install -q -U git+https://github.com/huggingface/accelerate.git
# pip install -q datasets

# module load python/3.9.0
# export TRANSFORMERS_CACHE=/home/groups/ramr/liutony/models/cache/

# Load dataset
import torch
#from datasets import load_dataset, load_from_disk
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

#data_test= load_from_disk("austin_test_noans.hf")
#data_test = data_test.train_test_split(test_size=0.30)
#data = load_dataset("Abirate/english_quotes")
#print(data_test)

adapter_path="out/checkpoint-84"     # input: adapters
save_to="models/Mistral-7B-finetuned-V2"    # out: merged model ready for inference

model_id = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)

# Add/set tokens (same 5 lines of code we used before training)
tokenizer.pad_token = tokenizer.eos_token

# Load LoRA adapter and merge
model_n = PeftModel.from_pretrained(base_model, adapter_path)
model_n = model_n.merge_and_unload()

model_n.save_pretrained(save_to, safe_serialization=True, max_shard_size='4GB')
tokenizer.save_pretrained(save_to)
