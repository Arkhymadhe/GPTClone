# -*- coding: utf-8 -*-

import torch
from torch import nn


class TokenEmbedder(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding_layer(x)


class EmbeddingSystem(nn.Module):
    def __init__(
        self,
        num_text_embeddings=1024,
        text_embedding_dim=128,
        num_pos_embeddings=1024,
        pos_embedding_dim=128,
    ):
        super().__init__()
        self.encoder_embedding = TokenEmbedder(num_text_embeddings, text_embedding_dim)
        self.encoder_pos_embedding = TokenEmbedder(
            num_pos_embeddings, pos_embedding_dim
        )

    def forward(self, x):
        x = self.encoder_embedding(x) + self.encoder_pos_embedding(
            torch.arange(0, x.shape[-1], 1).to(x.device)
        )
        return x
