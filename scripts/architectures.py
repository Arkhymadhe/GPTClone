import numpy as np
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


class Mask(nn.Module):
    def __init__(self, target_seq_len = 10, source_seq_len=10):
        super().__init__()
        self.source_seq_len = source_seq_len
        self.target_seq_len = target_seq_len

        self.mask = torch.full(size=(1, target_seq_len, source_seq_len), fill_value=-torch.inf)
        self.mask = torch.triu(self.mask, diagonal=1)

    @torch.no_grad()
    def forward(self, x):
        return x + self.mask.to(x.device)


class ALIBI(nn.Module):
    def __init__(self, mask=None, num_heads=8, seq_len=100):
        super(ALIBI, self).__init__()
        self.num_heads = num_heads
        self.seq_len = seq_len

        self.slopes = np.geomspace(
            start=2 ** (-8 / num_heads), stop=2 ** (-8), num=num_heads, endpoint=True
        )

        if mask is not None:
            self.mask = torch.zeros_like(mask)
        else:
            self.mask = torch.zeros(size=(num_heads, self.seq_len, self.seq_len))

        for slope, index in zip(self.slopes, range(self.seq_len)):
            if index < 2:
                continue
            sub_mask = torch.arange(index) - (index - 1)

            self.mask[:, index, :index] = slope * sub_mask

    @torch.no_grad()
    def forward(self, x):
        masked_x = x + self.mask.to(x.device)
        return masked_x


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
            state_dim=hidden_dim,
            narrow=narrow,
        )
        self.transform_vector = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 1.5), bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(
                in_features=int(hidden_dim * 1.5), out_features=hidden_dim, bias=False
            ),
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
            state_dim=hidden_dim,
        )
        self.cross_attention = MultiHeadAttention(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=hidden_dim,
        )
        self.transform_vector = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 1.5), bias=False
            ),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(
                in_features=int(hidden_dim * 1.5),
                out_features=num_embeddings,
                bias=False,
            ),
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

    def forward(
        self, x_encoder, x_decoder, source_mask=None, target_mask=None, train=True
    ):
        encoder_states = self.encoder(x_encoder, mask=source_mask)
        self.decoder.set_states(encoder_states, cross=True)

        if train:
            decoder_predictions = self.decoder(
                x_decoder,
                encoder_states,
                source_mask=source_mask,
                target_mask=target_mask,
            )
        else:
            while x_decoder.shape[-1] < self.prediction_len:
                decoder_predictions = self.decoder(
                    x_decoder,
                    encoder_states,
                    source_mask=source_mask,
                    target_mask=target_mask,
                )
                x_decoder = torch.cat(
                    [x_decoder, self.decode_tokens(decoder_predictions)], dim=-1
                )

        return decoder_predictions

    def decode_tokens(self, decoder_predictions):
        return torch.softmax(decoder_predictions, dim=-1).squeeze()


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        pre_ln=False,
    ):
        super().__init__()

        self.pre_ln = pre_ln

        self.attention = MultiHeadAttention(
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_heads=num_heads,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 4), bias=False
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=int(hidden_dim * 4), out_features=hidden_dim, bias=False
            ),
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x, source_mask=None):
        self.attention.set_states(x)

        if self.pre_ln:
            x = x + self.dropout1(self.attention(self.layer_norm1(x), mask=source_mask))
            x = x + self.dropout2(self.feed_forward(self.layer_norm2(x)))
        else:
            x = self.layer_norm1(x + self.dropout1(self.attention(x, mask=source_mask)))
            x = self.layer_norm2(x + self.dropout2(self.feed_forward(x)))

        return x


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        pre_ln=True,
        ablate=True,
    ):
        super().__init__()

        self.pre_ln = pre_ln

        self.ablate = ablate

        if not ablate:
            self.cross_attention = MultiHeadAttention(
                states=states,
                transform_states=transform_states,
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
            )
            self.layer_norm2 = nn.LayerNorm(hidden_dim)
            self.dropout2 = nn.Dropout(0.5)

        self.masked_attention = MultiHeadAttention(
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_heads=num_heads,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim, out_features=int(hidden_dim * 4), bias=False
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=int(hidden_dim * 4), out_features=hidden_dim, bias=False
            ),
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x, encoder_states, source_mask=None, target_mask=None):
        self.masked_attention.set_states(x)

        if not self.ablate:
            self.cross_attention.set_states(encoder_states)

        if self.pre_ln:
            x = x + self.dropout1(
                self.masked_attention(self.layer_norm1(x), mask=target_mask)
            )

            if not self.ablate:
                x = x + self.dropout2(
                    self.cross_attention(self.layer_norm2(x), mask=source_mask)
                )

            x = x + self.dropout3(self.feed_forward(self.layer_norm3(x)))
        else:
            x = self.layer_norm1(
                x + self.dropout1(self.masked_attention(x, mask=target_mask))
            )

            if not self.ablate:
                x = self.layer_norm2(
                    x + self.dropout2(self.cross_attention(x, mask=source_mask))
                )

            x = self.layer_norm3(x + self.dropout3(self.feed_forward(x)))

        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_encoder_blocks=4,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        pre_ln=False,
    ):
        super(TransformerEncoder, self).__init__()

        self.encoder = [
            TransformerEncoderBlock(
                states=states,
                transform_states=transform_states,
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                pre_ln=pre_ln,
            )
            for _ in range(num_encoder_blocks)
        ]
        self.encoder = nn.ModuleList(self.encoder)

    def forward(self, x_encoder, source_mask=None):
        for encoder_block in self.encoder:
            x_encoder = encoder_block(x_encoder, source_mask=source_mask)

        return x_encoder


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_decoder_blocks=4,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        ablate=True,
        pre_ln=False,
    ):
        super(TransformerDecoder, self).__init__()

        self.decoder = [
            TransformerDecoderBlock(
                states=states,
                transform_states=transform_states,
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                ablate=ablate,
                pre_ln=pre_ln,
            )
            for _ in range(num_decoder_blocks)
        ]

        self.decoder = nn.ModuleList(self.decoder)

    def forward(self, x_decoder, x_encoder, source_mask=None, target_mask=None):
        for decoder_block in self.decoder:
            x_decoder = decoder_block(
                x_decoder, x_encoder, source_mask=source_mask, target_mask=target_mask
            )

        return x_decoder


class Transformer(nn.Module):
    def __init__(
        self,
        num_encoder_blocks=4,
        num_decoder_blocks=4,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
        decoder_only=False,
        ablate=True,
        pre_ln=False,
    ):
        super(Transformer, self).__init__()

        self.decoder_only = decoder_only

        if not self.decoder_only:
            self.encoder = TransformerEncoder(
                states=states,
                transform_states=transform_states,
                narrow=narrow,
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_heads=num_heads,
                num_encoder_blocks=num_encoder_blocks,
                pre_ln=pre_ln,
            )

        self.decoder = TransformerDecoder(
            states=states,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=state_dim,
            num_heads=num_heads,
            num_decoder_blocks=num_decoder_blocks,
            ablate=ablate,
            pre_ln=pre_ln,
        )

    def forward(self, x_encoder, x_decoder, source_mask=None, target_mask=None):
        if not self.decoder_only:
            x_encoder = self.encoder(x_encoder, source_mask=source_mask)

        x_decoder = self.decoder(
            x_decoder, x_encoder, source_mask=source_mask, target_mask=target_mask
        )

        return x_decoder
