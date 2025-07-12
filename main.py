# Import necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer 
from sklearn.metrics import accuracy_score 
import torch

# Load AG news dataset
dataset = load_dataset("ag_news")

print(dataset["train"][0])
print(dataset["train"].features)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#tokenize function
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Prepare the dataset for PyTorch
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load pre-trained model with a classification head
# 4 labels in AG News (World, Sports, Business, Sci/Tech)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)

#split the dataset into train and test sets
train_data = tokenized_dataset["train"]
test_data = tokenized_dataset["test"]

# Define metrics function
def compute_metrics(pred):
    logits, labels = pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# Setup training arguments
training_args = TrainingArguments(
    output_dir="./results",
    do_eval=True,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()