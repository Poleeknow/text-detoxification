import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelWithLMHead
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

FINETUNE_PARAM = 1000
NUM_OF_EPOCHS = 2

# Class that is used to prepare the data for model
class MyDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_source_length, max_target_length):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.dataframe)
    
    def priint(self, idx):
        row = self.dataframe.iloc[idx]
        print(row["Toxic"])

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        inputs = self.tokenizer(
            row["Toxic"]
        )
        outputs = self.tokenizer(
            row["Neutral"]
        )
        inputs["input_ids"] = torch.tensor(inputs["input_ids"])
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"])
        inputs["labels"] = torch.tensor(outputs["input_ids"])
        return inputs


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Read in data from a CSV file
# data = pd.read_csv("data/raw/filtered.tsv", sep="\t", index_col=0)
# data.head(7)

# Load input data 
input_data = pd.read_csv("data/interim/filtered_for_models.csv", index_col=0)

train_set, val_set = train_test_split(input_data[:FINETUNE_PARAM], test_size=0.2, random_state=42)

# Load model and tokenizer from pretrained
tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir=None)
model = AutoModelWithLMHead.from_pretrained("t5-small", cache_dir=None)

data_collator = DataCollatorForSeq2Seq(tokenizer)

# %%
# Recollect data to fit for the model
train_dataset = MyDataset(train_set, tokenizer, 128, 128)
val_dataset = MyDataset(val_set, tokenizer, 128, 128)

training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=124,
    per_device_eval_batch_size=124,
    num_train_epochs=NUM_OF_EPOCHS,
    logging_dir='./logs',
    save_strategy="steps",
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=300,
    logging_steps=50,
    learning_rate=1e-4,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Fine tuning
trainer.train()

torch.save(model.state_dict(), "models/t5small.pt")





