import numpy as np
import torch
from attention import MultiHeadAttention
from architectures import Decoder, Encoder, EncoderDecoder, Transformer
from gpt import GPT

from torchinfo import summary

# Runnable tests
tests = [
    "all",
    "attention",
    "encoder",
    "decoder",
    "encdec",
    "transformer",
    "gpt1",
    "gpt2",
    "gpt3",
]

test_names = [
    "Attention modules",
    "Encoder architecture",
    "Decoder architecture",
    "Encoder-Decoder architecture",
    "Transformer architecture",
    "GPT-1 architecture",
    "GPT-2 architecture",
    "GPT-3 architecture",
]

test_map = {k: v for (k, v) in zip(tests[1:], test_names)}


def run_tests(test_to_run=None, device="cpu"):
    hidden_dim = 160
    num_heads = 16
    narrow = True
    transform_states = True

    x = torch.randn(size=(32, 5, hidden_dim)).to(device)
    query = torch.randn(size=(32, 10, hidden_dim)).to(device)

    if test_to_run == "encoder":
        print("Testing Encoder\n")

        attn = Encoder(
            num_heads=num_heads,
            narrow=narrow,
            transform_states=transform_states,
            hidden_dim=hidden_dim,
        ).to(device)

        ans = attn(query)

        print(summary(attn, input_data=query, device=device))

        # print("Attention scores shape: ", attn.attention_scores.shape)
        print("Encoder hidden states shape: ", ans.shape, end="\n\n")

    elif test_to_run == "attention":
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
            num_heads=num_heads,
            transform_states=transform_states,
            hidden_dim=hidden_dim,
            state_dim=hidden_dim,
            narrow=narrow,
            states=x,
        ).to(device)

        ans = attn(query)

        attn.get_attention_scores()

        print(summary(attn, input_data=query, device=device))

        print("Attention scores shape: ", attn.attention_scores.shape)
        print("Context vector shape: ", ans.shape, end="\n\n")

    elif test_to_run == "decoder":
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

        attn = Decoder(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
        ).to(device)

        attn.set_states(x, cross=True)

        ans = attn(query)

        print(summary(attn, input_data=query, device=device))

        print("Decoder predictions shape: ", ans.shape, end="\n\n")

    elif test_to_run == "encdec":
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

        enc = Encoder(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
        ).to(device)
        dec = Decoder(
            num_heads=num_heads,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
        ).to(device)

        attn = EncoderDecoder(encoder=enc, decoder=dec)

        ans = attn(x, query)

        print(summary(attn, input_data=[x, query], device=device))

        print("Encoder input shape: ", x.shape)
        print("Decoder input shape: ", query.shape)
        print("Decoder prediction shape: ", ans.shape, end="\n\n")

    elif test_to_run == "transformer":
        num_encoder_embeddings = 12
        num_decoder_embeddings = 200

        prob = torch.randn(size=(1,))

        if prob >= 0.5:
            print("Using Encoder-Decoder Transformer architecture...\n")
            decoder_only = False
        else:
            print("Using Transformer-Decoder architecture (No Encoder)...\n")
            decoder_only = True

        transformer = Transformer(
            vocab=num_decoder_embeddings,
            transform_states=transform_states,
            narrow=narrow,
            hidden_dim=hidden_dim,
            state_dim=hidden_dim,
            decoder_only=decoder_only,
        ).to(device)

        ans = transformer(x, query)

        print(
            summary(
                transformer,
                input_data=[x, query],
                depth=4,
                batch_dim=None,
                device=device,
            )
        )

        print("Transformer output shape: ", ans.shape)
        print("`decoder_only` : ", decoder_only)
        print("`transform_states`: ", transform_states)

    elif test_to_run in ["gpt1", "gpt2", "gpt3"]:
        if test_to_run == "gpt1":
            num_encoder_embeddings = 50257
            max_token = 1024
            narrow = True
            num_heads = 12
            num_blocks = 12
            embedding_dim = 768
        elif test_to_run == "gpt2":
            num_encoder_embeddings = 50257
            max_token = 2048
            narrow = True
            num_heads = 16
            num_blocks = 48
            embedding_dim = 1600
        elif test_to_run == "gpt3":
            num_encoder_embeddings = int(50257 / 3)
            max_token = 4096
            narrow = True
            num_heads = 12
            num_blocks = int(96 / 3)
            embedding_dim = int(12288 / 3)

        torch.cuda.empty_cache()

        x = torch.randint(
            low=0,
            high=num_encoder_embeddings,
            size=(
                1,
                1,
            ),
        ).to(device)

        gpt3 = GPT(
            transform_states=transform_states,
            num_heads=num_heads,
            num_decoder_blocks=num_blocks,
            embedding_dim=embedding_dim,
            num_embeddings=num_encoder_embeddings,
            max_token=max_token,
            narrow=narrow,
        ).to(device)

        ans = torch.sum(gpt3(x), dim=1)

        print(
            summary(
                gpt3,
                input_data=x,
                depth=4,
                batch_dim=None,
                device=device,
            )
        )

        print(f"{test_map[test_to_run].split(' ')[0]} prediction shape: ", ans.shape)

    return


if __name__ == "__main__":
    test = input("Which script to test: ").lower()

    while test not in tests:
        test = input(
            "Option provided must be in [`all`, `attention`, `encoder`, `decoder`, `encdec`, `transformer`, `gpt3`]:  "
        ).lower()

    device = input("\nWhich device to run on: ").lower()
    while device not in ["cuda", "cpu"]:
        device = input("Option provided must be in [`cuda`, `cpu`]:  ").lower()

    print(f"\n{device.upper()} device selected!\n")
    device = torch.device(device)

    if test != "all":
        run_tests(test_to_run=test, device=device)
    else:
        passed_tests = list()
        failed_tests = list()

        for test in tests[1:-1]:
            print("\n", "=" * 200)
            print(f"\nTesting {test_map[test]}...\n")
            try:
                run_tests(test_to_run=test, device=device)
                passed_tests.append(test)
            except:
                failed_tests.append(test)
                continue

        print("\n", "=" * 200)
        if len(passed_tests) < len(tests[1:]):
            print(
                f"Tests passed: {len(passed_tests)}/{len(passed_tests) + len(failed_tests)}\n"
            )

            print("Passed tests: ")
            for test in passed_tests:
                print(f" > {test}")

            if failed_tests:
                print("\nFailed tests: ")
                for test in failed_tests:
                    print(f" > {test}")
        else:
            print(" " * 90, "All tests run successfully!\n")
