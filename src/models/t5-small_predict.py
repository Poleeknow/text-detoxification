import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelWithLMHead
import nltk
nltk.download('punkt')
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

PUNCT_TO_REMOVE = string.punctuation
ENGLISH_STOPWORDS = set(stopwords.words("english"))

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


def paraphrase(text, max_length=128):

  input_ids = tokenizer.encode(text, return_tensors="pt", add_special_tokens=True)

  generated_ids = model.generate(input_ids=input_ids, num_return_sequences=5, num_beams=5, max_length=max_length, no_repeat_ngram_size=2, repetition_penalty=3.5, length_penalty=1.0, early_stopping=True)

  preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

  return preds


def text_to_tensor(sent):
    sent = sent.lower()
    sent = sent.translate(str.maketrans('', '', PUNCT_TO_REMOVE))
    sent = " ".join([word for word in str(sent).split() if word not in ENGLISH_STOPWORDS])
    sent = word_tokenize(sent)

    words = []
    for word in sent:
        query = list(vocabframe.query("key == @word")['translation'])
        if len(query) > 0:
            words.append(query[0])
    return torch.tensor(words, dtype=torch.int64)


def get_offset(sent):
    offset = [0]
    offset.append(sent.size(0))
    offset = torch.tensor(offset[:-1]).cumsum(dim=0)
    return offset

def predict(model, sent, offset):
    with torch.no_grad():
        model.eval()

        output = model(sent, offset)
        if output.item() > 1:
            score = 1.0
        else: score = output.item()

    return round(score, 4)


tokenizer = AutoTokenizer.from_pretrained("t5-small", cache_dir=None)
model = AutoModelWithLMHead.from_pretrained("t5-small", cache_dir=None)

cpt_t5 = torch.load("models/t5small.pt")
model.load_state_dict(cpt_t5)

input_data = pd.read_csv("data/interim/filtered_for_models.csv", index_col=0)
test_sample = input_data.sample(n=10)

paraphrased = []
for sent in test_sample['Toxic']:
    paraph = paraphrase(sent)
    paraphrased.append(paraph)

# Load vocabulary
vocabframe = pd.read_csv("data/interim/bestvocab.csv", index_col=0)

# Load metric model
metric_model = TextRegressionModel(114506, 300, 200)
cpt = torch.load("models/best.pt")
metric_model.load_state_dict(cpt)

# Score paraphpased sentences
predicted = []
pred_scores = []
for set5 in paraphrased:
    best_score = 1.1
    best = -1
    for i, sent in enumerate(set5):
        tokenized = text_to_tensor(sent)
        offset = get_offset(tokenized)
        score = predict(metric_model, tokenized, offset)
        if score < best_score:
            best_score = score
            best = i
    predicted.append(set5[best])
    pred_scores.append(best_score)

input_score = list(test_sample['Tox score'])
inputs = list(test_sample['Toxic'])

# Look at the scores
scores = pd.DataFrame(list(zip(inputs, input_score, pred_scores, predicted)), index=None, columns=['Toxic style', 'Before', 'After', 'Translation'])
print(scores.head())
