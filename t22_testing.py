from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "google/gemma-3-1b-it"  # Or your local path

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# Simulate NLI via prompting
def predict_nli_prompt(premise, hypothesis):
    prompt = f"""Premise: {premise}
Hypothesis: {hypothesis}
Label (entailment, neutral, contradiction):"""

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        label = decoded.split("Label")[-1].strip().split()[0].lower()
        return label

label = predict_nli_prompt("All dogs bark.", "Some dogs bark.")
print("Predicted label:", label)
