from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

token = "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"  # <-- Replace with your token

base_model_id = "google/gemma-3-1b-it"
adapter_path = "D:/University/LICENTA/gemma-3-1b-it-finetuned"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    token=token
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model = model.merge_and_unload()
model.save_pretrained("./merged_model")

tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=token)
tokenizer.save_pretrained("./merged_model")
