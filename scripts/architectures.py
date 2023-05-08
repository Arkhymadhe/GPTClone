import torch
from torch import nn

from attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_embeddings=1024,
        num_heads=32,
        narrow=True,
        transform_states=True,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            embedding_dim=embedding_dim, num_embeddings=num_embeddings
        )
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            transform_states=transform_states,
            hidden_dim=embedding_dim,
            narrow=narrow
        )
        self.transform_vector = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=int(embedding_dim * 1.5)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LazyLinear(out_features=embedding_dim),
        )
        return

    def forward(self, x, mask=None):
        embeddings = self.embedding_layer(x)
        self.self_attention.set_states(embeddings)

        attn_embeddings = self.self_attention(embeddings, mask=mask)
        new_attn_embeddings = self.transform_vector(attn_embeddings)

        return new_attn_embeddings


class Decoder(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        num_embeddings=1024,
        num_heads=32,
        narrow=True,
        transform_states=True,
    ):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )
        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=embedding_dim,
        )
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=embedding_dim,
        )
        self.transform_vector = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=int(embedding_dim * 1.5)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.LazyLinear(out_features=num_embeddings),
        )
        return

    def set_states(self, states, cross=True):
        if cross:
            self.cross_attention.set_states(states)
        else:
            self.self_attention.set_states(states)

        return

    def forward(self, x, source_mask=None, target_mask=None):
        embeddings = self.embedding_layer(x)
        self.self_attention.set_states(embeddings)

        attn_embeddings_1 = self.self_attention(embeddings, mask=target_mask)
        attn_embeddings_2 = self.cross_attention(attn_embeddings_1, mask=source_mask)

        decoder_predictions = self.transform_vector(attn_embeddings_2)

        return decoder_predictions


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_encoder, x_decoder):
        encoder_states = self.encoder(x_encoder)
        self.decoder.set_states(encoder_states, cross=True)

        decoder_predictions = self.decoder(x_decoder)

        return decoder_predictions