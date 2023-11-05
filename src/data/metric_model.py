import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.model_selection import train_test_split
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from torchmetrics.regression import R2Score

import torch.nn as nn

class TextRegressionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linearOut = nn.Linear(hidden_dim, 1)
        self.out = nn.Sigmoid()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        dout = self.dropout(embedded)
        lstm_out, (ht, ct) = self.lstm(dout)
        out = self.linear1(lstm_out)
        out = self.relu(out)
        out = self.linear1(out)
        out = self.relu(out)
        return self.linearOut(out)

ENGLISH_STOPWORDS = set(stopwords.words("english"))
PUNCT_TO_REMOVE = string.punctuation

def remove_stopwords(text: str) -> str:
    """custom function to remove the stopwords"""
    
    return " ".join([word for word in str(text).split() if word not in ENGLISH_STOPWORDS])

def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def tokenize_text(text: str) -> list[str]:
    return word_tokenize(text)

def yield_tokens(df):
    for sample1, sample2 in zip(df['reference_preprocess'], df['translation_preprocess']):
        sample = sample1 + sample2
        yield sample

def collate_batch(batch):
    target_list, text_list, similarity_list, offsets = [], [], [], [0]
    for _, _, similarity, _, target_r, target_t, reference, translation in batch:
        target_list.append(target_r)
        target_list.append(target_t)
        similarity_list.append(similarity)
        
        processed_text = torch.tensor(vocab(reference), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))

        processed_text = torch.tensor(vocab(translation), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
        
    target_list = torch.tensor(target_list)
    similarity_list = torch.tensor(similarity_list)
    text_list = torch.cat(text_list)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    return target_list.to(device), text_list.to(device), offsets.to(device), similarity_list.to(device)

def train_one_epoch(
    model,
    loader,
    optimizer,
    loss_fn,
    epoch_num=-1
):
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: train",
        leave=True,
    )
    model.train()
    train_loss = 0.0
    for i, batch in loop:
        targets, texts, offsets, similarities = batch
        targets = torch.reshape(targets, (-1, 1))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(texts, offsets)

        # loss calculation
        loss = loss_fn(outputs, targets)
        
        # backward pass
        loss.backward()

        # optimizer run
        optimizer.step()

        train_loss += loss.item()
        loop.set_postfix({"loss": float(loss)})

def val_one_epoch(
    model,
    loader,
    loss_fn,
    epoch_num=-1,
    best_so_far=0.0,
    ckpt_path='models/best.pt'
):
    
    loop = tqdm(
        enumerate(loader, 1),
        total=len(loader),
        desc=f"Epoch {epoch_num}: val",
        leave=True,
    )
    val_loss = 0.0
    correct = 0
    total = 0
    best_score = 0
    with torch.no_grad():
        model.eval()  # evaluation mode
        for i, batch in loop:
            targets, texts, offsets, similarities = batch
            targets = torch.reshape(targets, (-1, 1))

            # forward pass
            outputs = model(texts, offsets)
            
            loss = loss_fn(outputs, targets)
            
            r2score = R2Score()
            r2score = r2score(outputs, targets)

            val_loss += loss.item()
            loop.set_postfix({"loss": float(loss), "r2score": r2score.item()})

        if r2score > best_score:
            torch.save(model.state_dict(), ckpt_path)
            best_so_far = model.state_dict()
            best_score = r2score

    return best_so_far

data = pd.read_csv("filtered_paranmt/filtered.tsv", sep="\t", index_col=0)

data['reference_preprocess'] = data['reference'].str.lower()
data['translation_preprocess'] = data['translation'].str.lower()


data["reference_preprocess"] = data["reference_preprocess"].apply(lambda text: remove_stopwords(text))
data["translation_preprocess"] = data["translation_preprocess"].apply(lambda text: remove_stopwords(text))


data["reference_preprocess"] = data["reference_preprocess"].apply(lambda text: remove_punctuation(text))
data["translation_preprocess"] = data["translation_preprocess"].apply(lambda text: remove_punctuation(text))


data["reference_preprocess"] = data["reference_preprocess"].apply(lambda text: tokenize_text(text))
data["translation_preprocess"] = data["translation_preprocess"].apply(lambda text: tokenize_text(text))

train_split, val_split = train_test_split(data, test_size=0.2, random_state=420)

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

vocab = build_vocab_from_iterator(yield_tokens(train_split), specials=special_symbols)
vocab.set_default_index(UNK_IDX)

vocabframe = pd.DataFrame(vocab.get_stoi().items(), columns=['key', 'translation'])

pd.DataFrame.to_csv(vocabframe, "data/interim/bestvocab.csv")

train_dataloader = DataLoader(
    train_split.to_numpy(), batch_size=128, shuffle=True, collate_fn=collate_batch
)

val_dataloader = DataLoader(
    val_split.to_numpy(), batch_size=128, shuffle=False, collate_fn=collate_batch
)

epochs = 5
vocab_size = len(vocab)
embed_dim = 300
hidden_dim = 200
model = TextRegressionModel(vocab_size, embed_dim, hidden_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
model

best = -float('inf')
for epoch in range(epochs):
    train_one_epoch(model, train_dataloader, optimizer, loss_fn, epoch_num=epoch)
    best = val_one_epoch(model, val_dataloader, loss_fn, epoch, best_so_far=best)