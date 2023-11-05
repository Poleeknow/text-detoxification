import pandas as pd
import numpy as np
import re
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchtext.data.utils import get_tokenizer
import io
from collections import Counter
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import random
from typing import Tuple
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
import math
import time

N_EPOCHS = 1
CLIP = 1

class Encoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 emb_dim: int,
                 hid_dim: int,
                 n_layers: int,
                 dropout: float):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=1)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: Tensor) -> Tuple[Tensor]:
        # Ensure src is of type long
        src = src.long()

        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
    
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self,
                 output_dim: int,
                 emb_dim: int,
                 hid_dim: int,
                 n_layers: int,
                 dropout: int
                 ):
        super().__init__()

        self.output_dim = output_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=self.n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: Tensor, hidden: Tensor, cell: Tensor) -> Tuple[Tensor]:

        input = input.unsqueeze(0)

        embedded = self.dropout(self.embedding(input))

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        prediction = self.fc_out(output.squeeze(0))

        return prediction, hidden, cell
    

class Seq2Seq(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 device: torch.device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            'hidden dimensions of encoder and decoder must be equal.'
        assert encoder.n_layers == decoder.n_layers, \
            'n_layers of encoder and decoder must be equal.'

    def forward(self, src: Tensor, trg: Tensor, teacher_forcing_ratio: float = 0.5) -> Tensor:

        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> token
        input = trg[0,:]

        for t in range(1, max_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            output = trg[t] if teacher_force else top1

        return outputs


def train(model: nn.Module,
          iterator: DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0

    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0)

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def generate_batch(data_batch, PAD_IDX, BOS_IDX, EOS_IDX):
    toxic_batch, neutral_batch = [], []
    for (t_item, n_item) in data_batch:
        toxic_batch.append(torch.cat([torch.tensor([BOS_IDX]), t_item, torch.tensor([EOS_IDX])], dim=0))
        neutral_batch.append(torch.cat([torch.tensor([BOS_IDX]), n_item, torch.tensor([EOS_IDX])], dim=0))
    toxic_batch = pad_sequence(toxic_batch, padding_value=PAD_IDX)
    neutral_batch = pad_sequence(neutral_batch, padding_value=PAD_IDX)
    return toxic_batch, neutral_batch

def build_vocab(data, tokenizer):
    counter = Counter()
    for text in data:
        counter.update(tokenizer(text))
    return vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])

def data_process(data):
    raw_toxic_iter = data['Toxic']
    raw_neutral_iter = data['Neutral']
    data = []
    for (raw_de, raw_en) in zip(raw_toxic_iter, raw_neutral_iter):
        toxic_tensor = torch.tensor([toxic_vocab[token] for token in en_tokenizer(raw_de)],
                                dtype=torch.long)
        neutral_tensor = torch.tensor([neutral_vocab[token] for token in en_tokenizer(raw_en)],
                                dtype=torch.long)
        data.append((toxic_tensor, neutral_tensor))
    return data

def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

input_data = pd.read_csv("data/interim/filtered_for_models.csv", index_col=0)

train_set, val_set = train_test_split(input_data[:10000], test_size=0.2, random_state=42)

en_tokenizer = get_tokenizer('spacy', language='en')

toxic_vocab = build_vocab(train_set['Toxic'], en_tokenizer)
toxic_vocab.set_default_index(0)
neutral_vocab = build_vocab(train_set['Neutral'], en_tokenizer)
neutral_vocab.set_default_index(0)

train_data = data_process(train_set)
val_data = data_process(val_set)


BATCH_SIZE = 128
PAD_IDX = toxic_vocab['<pad>']
BOS_IDX = toxic_vocab['<bos>']
EOS_IDX = toxic_vocab['<eos>']


train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)
valid_iter = DataLoader(val_data, batch_size=BATCH_SIZE,
                        shuffle=True, collate_fn=generate_batch)


INPUT_DIM = len(toxic_vocab)
OUTPUT_DIM = len(neutral_vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

encoder = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)

decoder = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(encoder, decoder, device).to(device)

model.apply(init_weights)

print(f'The model has {count_parameters(model):,} trainable parameters')

optimizer = optim.Adam(model.parameters())

PAD_IDX = neutral_vocab.get_stoi()['<pad>']

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/Seq2SeqModel.pt')

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')