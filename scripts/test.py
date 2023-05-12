import numpy as np
import torch
from attention import MultiHeadAttention
from architectures import Decoder, Encoder, EncoderDecoder, Transformer

from torchinfo import summary

if __name__ == "__main__":
    test = input("Which script to test: ").lower()

    while test not in ["attention", "encoder", "decoder", "encdec", "transformer"]:
        test = input(
            "Option provided must be in [`attention`, `encoder`, `decoder`, `encdec`, `transformer`]:  "
        ).lower()

    x = torch.randn(size=(32, 5, 128))
    query = torch.randn(size=(32, 10, 128))

    num_heads = 16
    narrow = False
    transform_states = False

    if test == "encoder":
        print("Testing Encoder\n")

        attn = Encoder(num_heads=num_heads, narrow=narrow, transform_states=transform_states)

        ans = attn(query)

        # print("Attention scores shape: ", attn.attention_scores.shape)
        print("Context vector shape: ", ans.shape, end="\n\n")

        print(summary(attn, input_data=query, device="cpu"))

    elif test == "attention":
        print("Testing attention...\n")

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
            num_heads=num_heads, transform_states=transform_states, narrow=narrow, states=x
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

        print(summary(attn, input_data=query, device="cpu"))

    elif test == "decoder":
        print("Testing Decoder\n")

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

        attn = Decoder(num_heads=num_heads, transform_states=transform_states, narrow=narrow)
        attn.set_states(x, cross=True)

        ans = attn(query)

        print("Context vector shape: ", ans.shape, end="\n\n")

        print(summary(attn, input_data=query, device="cpu"))

    elif test == "encdec":
        print("Testing Encoder-Decoder architecture\n")

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

        enc = Encoder(num_heads=num_heads, transform_states=transform_states, narrow=narrow)
        dec = Decoder(num_heads=num_heads, transform_states=transform_states, narrow=narrow)

        attn = EncoderDecoder(encoder=enc, decoder=dec)

        ans = attn(x, query)

        print("Encoder input shape: ", x.shape)
        print("Decoder input shape: ", query.shape)
        print("Decoder prediction shape: ", ans.shape, end="\n\n")

        print(summary(attn, input_data=[x, query], device="cpu"))

    elif test == "transformer":
        num_encoder_embeddings = 12
        num_decoder_embeddings = 200

        prob = torch.randn(size=(1,))

        if prob >= 0.5:
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
            print("Using Transformer-Decoder architecture (No Encoder)...\n")
            decoder_only = True

        query = torch.randint(
            low=0,
            high=num_decoder_embeddings,
            size=(
                32,
                10,
            ),
        )

        transformer = Transformer(
            num_decoder_embeddings=num_decoder_embeddings,
            num_encoder_embeddings=num_encoder_embeddings,
            transform_states=transform_states,
            decoder_only=decoder_only,
        )

        ans = transformer(x, query)

        print(ans.shape)

        print(
            summary(
                transformer,
                input_data=[x, query],
                depth=2,
                batch_dim=None,
                device="cpu",
            )
        )

        print("`decoder_only` : ", decoder_only)
        print("`transform_states`: ", transform_states)
