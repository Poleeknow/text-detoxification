
import pandas as pd
import numpy as np
import re
import torch
import string 


PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def create_inputs(dataset):
    toxic_set = []
    neutral_set = []
    tox_scores = []
    for ref, trn, ref_score, trn_score in zip(dataset['reference'], dataset['translation'], dataset['ref_tox'], dataset['trn_tox']):
        if ref_score > 0.6 and trn_score < 0.4:
            tox_scores.append(ref_score)
            toxic_set.append(ref)
            neutral_set.append(trn)
        if ref_score < 0.4 and trn_score > 0.6:
            tox_scores.append(trn_score)
            toxic_set.append(trn)
            neutral_set.append(ref)
    
    toxic_set = [text.lower() for text in toxic_set]
    toxic_set = [remove_punctuation(text) for text in toxic_set]

    neutral_set = [text.lower() for text in neutral_set]
    neutral_set = [remove_punctuation(text) for text in neutral_set]
    return toxic_set, neutral_set, tox_scores


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Read in data from a CSV file
data = pd.read_csv("data/raw/filtered.tsv", sep="\t", index_col=0)
print(data.head(7))

toxic_set, neutral_set, tox_scores = create_inputs(data)

input_data = pd.DataFrame(list(zip(toxic_set, neutral_set, tox_scores)), columns=['Toxic', 'Neutral', 'Tox score'])

pd.DataFrame.to_csv(input_data, "data/interim/filtered_for_models.csv")



