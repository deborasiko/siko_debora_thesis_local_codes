# attention_visualizer_gemma.py

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load Gemma (Causal LM only) ===
model_path = "google/gemma-3-1b-it"  # Replace with local path if needed
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, output_attentions=True)
model.eval()

# === Attention visualization for a single prompt ===
def show_attention(prompt, layer=-1):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    attentions = outputs.attentions  # List: [layer_1, layer_2, ..., layer_n]
    attn_matrix = attentions[layer][0]  # Shape: [heads, seq_len, seq_len]
    avg_attn = attn_matrix.mean(dim=0).detach().numpy()

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_attn, xticklabels=tokens, yticklabels=tokens, cmap="Blues", square=True)
    plt.title(f"Gemma Attention Map (Layer {layer})")
    plt.tight_layout()
    plt.show()

# === Prompt-style input ===
premise = "All dogs bark."
hypothesis = "No dogs bark."
prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nLabel:"

print("Visualizing attention for:\n", prompt)
show_attention(prompt)
