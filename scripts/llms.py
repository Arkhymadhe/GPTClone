import numpy as np
import torch
from torch import nn

from architectures import Transformer, EmbeddingSystem


class GPT(nn.Module):
    def __init__(
        self,
        states=None,
        num_heads=96,
        num_embeddings=50257,
        max_token=2048,
        embedding_dim=12288,
        num_decoder_blocks=96,
        narrow=True,
        pre_ln=False,
        transform_states=True,
        ablate=True
    ):
        super(GPT, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_decoder_blocks = num_decoder_blocks
        self.transform_states = transform_states
        self.narrow = narrow
        self.states = states
        self.num_heads = num_heads

        self.embedding_system = EmbeddingSystem(
            text_embedding_dim=embedding_dim,
            pos_embedding_dim=embedding_dim,
            num_text_embeddings=num_embeddings,
            num_pos_embeddings=max_token,
        )

        self.decoder = Transformer(
            states=self.states,
            num_decoder_blocks=self.num_decoder_blocks,
            transform_states=self.transform_states,
            decoder_only=True,
            narrow=self.narrow,
            hidden_dim=self.embedding_dim,
            state_dim=self.embedding_dim,
            num_heads=self.num_heads,
            ablate=ablate,
            pre_ln=pre_ln
        )

    def forward(self, x, source_mask=None, target_mask=None):
        x_new = self.embedding_system(x)

        x_new = self.decoder(
            x_new, x_new, source_mask=source_mask, target_mask=target_mask
        )

        return torch.log_softmax(x_new, dim=-1)


class GPTHead(nn.Module):
    def __init__(self, hidden_dim=128, vocab=50257):
        super(GPTHead, self).__init__()
        self.decoder_head = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 4), bias=False
            ),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_dim * 4), out_features=vocab, bias=False),
        )

    def forward(self, x):
        x = self.decoder_head(x)
        return x


class BLOOM(nn.Module):
    def __init__(self,
        states=None,
        num_heads=96,
        num_embeddings=50257,
        max_token=2048,
        embedding_dim=12288,
        num_decoder_blocks=96,
        narrow=True,
        pre_ln=False,
        transform_states=True,
        ablate=True):
        super(BLOOM, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.num_decoder_blocks = num_decoder_blocks
        self.transform_states = transform_states
        self.narrow = narrow
        self.states = states
        self.num_heads = num_heads

        self.input_embedding_system = EmbeddingSystem(
            text_embedding_dim=embedding_dim,
            pos_embedding_dim=embedding_dim,
            num_text_embeddings=num_embeddings,
            num_pos_embeddings=max_token,
        )
        self.input_layer_norm = nn.LayerNorm(embedding_dim)

        self.output_layer_norm = nn.LayerNorm(embedding_dim)

        self.decoder = Transformer(
            states=self.states,
            num_decoder_blocks=self.num_decoder_blocks,
            transform_states=self.transform_states,
            decoder_only=True,
            narrow=self.narrow,
            hidden_dim=self.embedding_dim,
            state_dim=self.embedding_dim,
            num_heads=self.num_heads,
            ablate=ablate,
            pre_ln=pre_ln
        )
    def forward(self, x, source_mask=None, target_mask=None):
        x_new = self.input_embedding_system(x)
        x_new = self.input_layer_norm(x_new)

        x_new = self.decoder(x_new, x_new, source_mask=source_mask, target_mask=target_mask)
        x_new = self.output_layer_norm(x_new)
        #x_new = self.output_embedding_system(x_new)

        return x_new


class BLOOMHead(nn.Module):
    def __init__(self):
        super(BLOOMHead, self).__init__()
        raise NotImplementedError
    def forward(self, x):
        raise NotImplementedError


class ALIBI(nn.Module):
    def __init__(self, num_heads=8, seq_len=100):
        super(ALIBI, self).__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len

        self.slopes = np.geomspace(
            start = 2 ** (-8/num_heads),
            stop = 2 ** (-8),
            num = num_heads
        )

        self.mask = torch.zeros(size=(num_heads, self.seq_len, self.seq_len))

        for slope, index in zip(self.slopes, range(self.seq_len)):
            if index < 2:
                continue
            sub_mask = torch.arange(index) - (index - 1)

            self.mask[:, index, :index] = slope * sub_mask

    @torch.no_grad()
    def forward(self, x):
        masked_x = x + self.mask
        return masked_x


