import torch
import pandas as pd
#import tqdm
from transformers.pipelines.pt_utils import KeyDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset, load_from_disk

# module load py-pandas
model_name_or_path = "./models/Mistral-7B-finetuned"
#model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
# To use a different branch, change revision
# For example: revision="gptq-4bit-64g-actorder_True"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             torch_dtype=torch.bfloat16, 
                                             low_cpu_mem_usage=True,
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=2000,
    do_sample=True,
    return_full_text=False,
    temperature=0.1,
    top_p=0.95,
    top_k=40,
    repetition_penalty=1.1
)

data = load_from_disk("austin_test_noans.hf")
data = data.train_test_split(test_size=0.30)


out_list = []
tokenizer.pad_token = tokenizer.eos_token
for out in pipe(KeyDataset(data["test"],"zeroprompt"), batch_size=1):
    #print(out)
    out_list.append(out[0]["generated_text"])
df = pd.DataFrame(out_list)
df.to_pickle('finetune_test.df')