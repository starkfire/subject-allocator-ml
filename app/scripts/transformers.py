from transformers import DistilBertModel, DistilBertTokenizer
import torch


def create_embedding(word: str,
                     model: DistilBertModel,
                     tokenizer: DistilBertTokenizer):

    inputs = tokenizer(word, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().flatten().tolist()

    return embedding


