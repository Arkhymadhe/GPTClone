import numpy as np
import torch
from attention import Attention, MultiHeadAttention
from architectures import Decoder, Encoder

if __name__ == "__main__":

    test = input("Which script to test: ")

    if test.lower() == "attention":
        x = torch.randn(size=(32, 5, 128))
        query = torch.randint(size=(32, 10), low=0, high=10)

        attn = Decoder(num_heads=2, narrow=False, transform_states=True)
        attn.set_states(states=x, cross=True)

        if hasattr(attn.self_attention.attention_heads[0], "keys_mlp"):
            print(True)
        else:
            print(False)

        ans = attn(query)

        # print("Attention scores shape: ", attn.attention_scores.shape)
        print("Context vector shape: ", ans.shape, end="\n\n")

        for name, param in attn.named_parameters():
            print(name, " : ", param.shape)
    else:
        x = torch.randn(size=(32, 5, 128))
        query = torch.randn(size=(32, 10, 128))

        num_heads = 5

        prob = torch.randint(low=0, high=11, size=(1,)) / 10

        if prob >= .5:
            transform_states = [
                np.random.choice([True, False]) for _ in range(num_heads)
            ]
        else:
            inner_prob = np.random.randint(low=0, high=11, size=(1,)) / 10
            if inner_prob >= .5:
                transform_states = True
            else:
                transform_states = False

        attn = MultiHeadAttention(num_heads=num_heads, transform_states=False, narrow=True, states=x)

        ans = attn(x)

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
