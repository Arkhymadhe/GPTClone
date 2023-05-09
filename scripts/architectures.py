import torch
from torch import nn

from attention import MultiHeadAttention


class TokenEmbedder(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=128):
        super().__init__()
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.embedding_layer(x)


class Encoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_heads=32,
        narrow=True,
        transform_states=True,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            transform_states=transform_states,
            hidden_dim=hidden_dim,
            narrow=narrow,
        )
        self.transform_vector = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim * 1.5)),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(in_features=int(hidden_dim * 1.5), out_features=hidden_dim),
        )
        return

    def forward(self, x, mask=None):
        self.self_attention.set_states(x)

        attn_embeddings = self.self_attention(x, mask=mask)
        new_attn_embeddings = self.transform_vector(attn_embeddings)

        return new_attn_embeddings


class Decoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_embeddings=1024,
        num_heads=32,
        narrow=True,
        transform_states=True,
    ):
        super().__init__()

        self.self_attention = MultiHeadAttention(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
        )
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
        )
        self.transform_vector = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim * 1.5)),
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

    def forward(self, x, encoder_states=None, source_mask=None, target_mask=None):
        self.set_states(x, cross=False)
        if encoder_states is not None:
            self.set_states(encoder_states, cross=True)

        attn_embeddings_1 = self.self_attention(x, mask=target_mask)
        attn_embeddings_2 = self.cross_attention(attn_embeddings_1, mask=source_mask)

        decoder_predictions = self.transform_vector(attn_embeddings_2)

        return decoder_predictions


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, prediction_len=20):
        super().__init__()
        self.prediction_len = prediction_len
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x_encoder, x_decoder, train=True):
        encoder_states = self.encoder(x_encoder)
        self.decoder.set_states(encoder_states, cross=True)

        if train:
            decoder_predictions = self.decoder(x_decoder, encoder_states)
        else:
            while x_decoder.shape[-1] < self.prediction_len:
                decoder_predictions = self.decoder(x_decoder, encoder_states)
                x_decoder = torch.cat(
                    [x_decoder, self.decode_tokens(decoder_predictions)], dim=-1
                )

        return decoder_predictions

    def decode_tokens(self, decoder_predictions):
        return torch.softmax(decoder_predictions, dim=-1).squeeze()


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_embeddings=1024,
        num_decoder_embeddings=1024,
        num_encoder_blocks=4,
        num_decoder_blocks=4,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        num_heads=32,
    ):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            num_encoder_embeddings=num_encoder_embeddings,
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_encoder_blocks=num_encoder_blocks,
        )

        self.decoder = TransformerDecoder(
            num_decoder_embeddings=num_decoder_embeddings,
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_decoder_blocks=num_decoder_blocks,
        )

        self.decoder_head = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim * 1.5)),
            nn.ReLU(),
            nn.Linear(
                in_features=int(hidden_dim * 1.5), out_features=num_decoder_embeddings
            ),
        )

    def forward(self, x_encoder, x_decoder, source_mask=None, target_mask=None):
        x_encoder = self.encoder(x_encoder, source_mask=source_mask)

        x_decoder = self.decoder(
            x_decoder, x_encoder, source_mask=source_mask, target_mask=target_mask
        )

        x_decoder = self.decoder_head(x_decoder)

        return x_decoder


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        num_heads=32,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim * 1.5)),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_dim * 1.5), out_features=hidden_dim),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, source_mask=None):
        self.attention.set_states(x)
        x = self.layer_norm1(x + self.attention(x, mask=source_mask))

        x = self.layer_norm2(self.feed_forward(x) + x)

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        num_heads=32,
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.masked_attention = MultiHeadAttention(
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=int(hidden_dim * 1.5)),
            nn.ReLU(),
            nn.Linear(in_features=int(hidden_dim * 1.5), out_features=hidden_dim),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x, encoder_states, source_mask=None, target_mask=None):
        self.attention.set_states(encoder_states)
        self.masked_attention.set_states(x)

        x = self.layer_norm1(x + self.masked_attention(x, mask=target_mask))

        x = self.layer_norm2(x + self.attention(x, mask=source_mask))

        x = self.layer_norm3(self.feed_forward(x) + x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_embeddings=1024,
        num_encoder_blocks=4,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        num_heads=32,
    ):
        super(TransformerEncoder, self).__init__()

        self.encoder_embedding = TokenEmbedder(num_encoder_embeddings, hidden_dim)
        self.encoder_pos_embedding = TokenEmbedder(num_encoder_embeddings, hidden_dim)

        self.encoder = [
            TransformerEncoderBlock(
                states=states,
                transform_states=transform_states,
                narrow=narrow,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(num_encoder_blocks)
        ]
        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, x_encoder, source_mask=None):
        x_encoder = self.encoder_embedding(x_encoder) + self.encoder_pos_embedding(
            x_encoder
        )
        for encoder_block in self.encoder:
            x_encoder = encoder_block(x_encoder, source_mask=source_mask)

        return x_encoder


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_decoder_embeddings=1024,
        num_decoder_blocks=4,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        num_heads=32,
    ):
        super(TransformerDecoder, self).__init__()

        self.decoder_embedding = TokenEmbedder(num_decoder_embeddings, hidden_dim)
        self.decoder_pos_embedding = TokenEmbedder(num_decoder_embeddings, hidden_dim)

        self.decoder = [
            TransformerDecoderBlock(
                states=states,
                transform_states=transform_states,
                narrow=narrow,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
            )
            for _ in range(num_decoder_blocks)
        ]

        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x_decoder, x_encoder, source_mask=None, target_mask=None):
        x_decoder = self.decoder_embedding(x_decoder) + self.decoder_pos_embedding(
            x_decoder
        )
        for decoder_block in self.decoder:
            x_decoder = decoder_block(
                x_decoder, x_encoder, source_mask=source_mask, target_mask=target_mask
            )

        return x_decoder
