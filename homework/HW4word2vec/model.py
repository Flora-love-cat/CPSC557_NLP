import torch
import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Word2Vec, self).__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.context_embed = nn.Embedding(vocab_size, embed_size)
        self.output = nn.LogSigmoid()

    def forward(self, target_word, context_word):
        # use `torch.bmm` for batch matrix multiplication
        target_embed = self.word_embed(target_word).unsqueeze(2)  # Shape: [batch_size, embed_size, 1]
        context_embed = self.context_embed(context_word).unsqueeze(1)  # Shape: [batch_size, 1, embed_size]
        scores = torch.bmm(context_embed, target_embed).squeeze()  # Shape: [batch_size]

        loglogits = self.output(scores)
        return loglogits