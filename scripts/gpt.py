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
    ):
        super().__init__()

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
            pre_ln=pre_ln
        )

    def forward(self, x, source_mask=None, target_mask=None):
        x_new = self.embedding_system(x)

        x_new = self.decoder(
            x_new, x_new, source_mask=source_mask, target_mask=target_mask
        )

        return torch.log_softmax(x_new, dim=-1)
