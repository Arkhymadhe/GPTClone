import numpy as np
import torch
from attention import Attention, MultiHeadAttention
from architectures import Decoder, Encoder, EncoderDecoder, Transformer

from torchinfo import summary

if __name__ == "__main__":
    test = input("Which script to test: ")

    assert test.lower() in [
        "attention",
        "encoder",
        "decoder",
        "encdec",
        "transformer",
    ], """ `test` must be one of "attention", "encoder", "decoder", "encdec", "transformer"."""

    if test.lower() == "encoder":
        print("Testing Encoder\n")
        x = torch.randn(size=(32, 5, 128))
        query = torch.randn(size=(32, 10, 128))

        attn = Encoder(num_heads=2, narrow=False, transform_states=True)

        ans = attn(query)

        # print("Attention scores shape: ", attn.attention_scores.shape)
        print("Context vector shape: ", ans.shape, end="\n\n")

        for name, param in attn.named_parameters():
            print(name, " : ", param.shape)

    elif test.lower() == "attention":
        print("Testing attention...\n")

        x = torch.randn(size=(32, 5, 128))
        query = torch.randn(size=(32, 10, 128))

        num_heads = 16

        prob = torch.randint(low=0, high=11, size=(1,)) / 10

        if prob >= 0.5:
            transform_states = [
                np.random.choice([True, False]) for _ in range(num_heads)
            ]
        else:
            inner_prob = torch.randint(low=0, high=11, size=(1,)) / 10
            if inner_prob >= 0.5:
                transform_states = True
            else:
                transform_states = False

        attn = MultiHeadAttention(
            num_heads=num_heads, transform_states=False, narrow=True, states=x
        )

        ans = attn(query)

        if hasattr(attn.attention_heads[0], "keys_mlp"):
            print(True)
        else:
            print(False)

        attn.get_attention_scores()

        print("Attention scores shape: ", attn.attention_scores.shape)
        print("Context vector shape: ", ans.shape, end="\n\n")

        print(attn.transform_states)

        for name, param in attn.named_parameters():
            print(name, " : ", param.shape)

    elif test.lower() == "decoder":
        print("Testing Decoder\n")

        x = torch.randn(size=(32, 5, 128))
        query = torch.randn(size=(32, 10, 128))

        num_heads = 5

        prob = torch.randint(low=0, high=11, size=(1,)) / 10

        if prob >= 0.5:
            transform_states = [
                np.random.choice([True, False]) for _ in range(num_heads)
            ]
        else:
            inner_prob = np.random.randint(low=0, high=11, size=(1,)) / 10
            if inner_prob >= 0.5:
                transform_states = True
            else:
                transform_states = False

        attn = Decoder(num_heads=num_heads, transform_states=False, narrow=True)
        attn.set_states(x, cross=True)

        ans = attn(query)

        print("Context vector shape: ", ans.shape, end="\n\n")

        for name, param in attn.named_parameters():
            print(name, " : ", param.shape)

    elif test.lower() == "encdec":
        print("Testing Encoder-Decoder architecture\n")

        x = torch.randn(size=(32, 5, 128))
        query = torch.randn(size=(32, 10, 128))

        num_heads = 5

        prob = torch.randint(low=0, high=11, size=(1,)) / 10

        if prob >= 0.5:
            transform_states = [
                np.random.choice([True, False]) for _ in range(num_heads)
            ]
        else:
            inner_prob = np.random.randint(low=0, high=11, size=(1,)) / 10
            if inner_prob >= 0.5:
                transform_states = True
            else:
                transform_states = False

        enc = Encoder(num_heads=num_heads, transform_states=True, narrow=False)
        dec = Decoder(num_heads=num_heads, transform_states=False, narrow=False)

        attn = EncoderDecoder(encoder=enc, decoder=dec)

        ans = attn(x, query)

        print("Encoder input shape: ", x.shape)
        print("Decoder input shape: ", query.shape)
        print("Decoder prediction shape: ", ans.shape, end="\n\n")

        for name, param in attn.named_parameters():
            print(name, " : ", param.shape)

    elif test.lower() == "transformer":
        num_encoder_embeddings = 12
        num_decoder_embeddings = 200

        prob = torch.randn(size=(1,))

        if prob >= .5:
            print("Using Encoder-Decoder Transformer architecture...\n")
            x = torch.randint(
                low=0,
                high=num_encoder_embeddings,
                size=(
                    32,
                    5,
                ),
            )
            decoder_only = False
        else:
            print("Using Transformer-Encoder architecture (No Encoder)...\n")
            x = torch.randn(size=(32, 5, 128))
            decoder_only = True

        query = torch.randint(
            low=0,
            high=num_decoder_embeddings,
            size=(
                32,
                10,
            ),
        )

        transform_states = True

        transformer = Transformer(
            num_decoder_embeddings=num_decoder_embeddings,
            num_encoder_embeddings=num_encoder_embeddings,
            transform_states=transform_states,
            decoder_only=decoder_only
        )

        ans = transformer(x, query)

        print(ans.shape)

        print(
            summary(
                transformer,
                input_data=[x, query],
                depth=6,
                batch_dim=None,
                device="cpu",
            )
        )

        print("`decoder_only` : ", decoder_only)
        print("`transform_states`: ", transform_states)
