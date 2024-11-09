from transformers import DistilBertModel, DistilBertTokenizer
import torch
from typing import List


def create_embedding(word: str | List[str],
                     model: DistilBertModel,
                     tokenizer: DistilBertTokenizer):

    inputs = tokenizer(word, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().flatten().tolist()

    return embedding


