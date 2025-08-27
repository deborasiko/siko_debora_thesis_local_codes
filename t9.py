#step 1
import pandas as pd

# Load the SICK dataset
df = pd.read_csv("SICK_train.txt", sep="\t")

# Keep only needed columns
df = df[["sentence_A", "sentence_B", "entailment_judgment"]]

# Drop missing labels (if any)
df = df.dropna(subset=["entailment_judgment"])

# Map labels to integers
label_map = {"ENTAILMENT": 0, "NEUTRAL": 1, "CONTRADICTION": 2}
df["label"] = df["entailment_judgment"].map(label_map)

#step 2
from datasets import Dataset
dataset = Dataset.from_pandas(df[["sentence_A", "sentence_B", "label"]])
dataset = dataset.train_test_split(test_size=0.1)


#step 3
def format_nli(example):
    return {
        "text": f"Premise: {example['sentence_A']}\nHypothesis: {example['sentence_B']}\nAnswer:",
        "label": example["label"]
    }

dataset = dataset.map(format_nli)

#step 4
from transformers import AutoTokenizer

model_id = "google/gemma-1.1-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(example):
    enc = tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
    enc["labels"] = example["label"]
    return enc

tokenized_dataset = dataset.map(tokenize, batched=True)

#step 5
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # May vary depending on Gemma's internal layers
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

#step 6
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./gemma-nli-sick",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    learning_rate=2e-4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer
)

trainer.train()

#step 7
model.save_pretrained("./gemma_sick_lora")
tokenizer.save_pretrained("./gemma_sick_lora")
