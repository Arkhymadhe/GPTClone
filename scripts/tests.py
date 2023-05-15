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
    x = torch.randn(size=(32, 5, 128)).to(device)
    query = torch.randn(size=(32, 10, 128)).to(device)

    num_heads = 16
    narrow = True
    transform_states = True

    if test_to_run == "encoder":
        print("Testing Encoder\n")

        attn = Encoder(
            num_heads=num_heads, narrow=narrow, transform_states=transform_states
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
            narrow=narrow,
            states=x,
        ).to(device)

        ans = attn(query)

        if hasattr(attn.attention_heads[0], "keys_mlp"):
            print(True)
        else:
            print(False)

        attn.get_attention_scores()

        print(summary(attn, input_data=query, device=device))

        print("Attention scores shape: ", attn.attention_scores.shape)
        print("Context vector shape: ", ans.shape, end="\n\n")

        print(attn.transform_states)


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
            num_heads=num_heads, transform_states=transform_states, narrow=narrow
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
            num_heads=num_heads, transform_states=transform_states, narrow=narrow
        ).to(device)
        dec = Decoder(
            num_heads=num_heads, transform_states=transform_states, narrow=narrow
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
            x = torch.randint(
                low=0,
                high=num_encoder_embeddings,
                size=(
                    32,
                    5,
                ),
            ).to(device)
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
        ).to(device)

        transformer = Transformer(
            num_decoder_embeddings=num_decoder_embeddings,
            num_encoder_embeddings=num_encoder_embeddings,
            transform_states=transform_states,
            narrow=narrow,
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
            narrow = True
            num_heads = 12
            num_blocks = 12
            embedding_dim = 768
        elif test_to_run == "gpt2":
            num_encoder_embeddings = int(50257/5)
            narrow = True
            num_heads = 12
            num_blocks = 48
            embedding_dim = 1600
        elif test_to_run == "gpt3":
            num_encoder_embeddings = int(50257/5)
            narrow = True
            num_heads = 12
            num_blocks = 96
            embedding_dim = int(12288/5)

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
            narrow=narrow,
        ).to(device)

        ans = torch.sum(gpt3(x), dim=1)

        print(
            summary(
                gpt3.decoder,
                input_data=x,
                depth=4,
                batch_dim=None,
                device=device,
            )
        )

        gpt3.decoder.decoder[0].masked_attention.get_attention_scores()
        print(gpt3.decoder.decoder[0].masked_attention.attention_scores)

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
        try:
            run_tests(test_to_run=test, device=device)
            print(f"\nTest `{test}` succeeded!")
        except:
            print(f"\nTest `{test}` failed.")
    else:
        passed_tests = list()
        failed_tests = list()

        for test in tests[1:]:
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
            print(f"Tests passed: {len(passed_tests)}/{len(tests[1:])}\n")

            print("Passed tests: ")
            for test in passed_tests:
                print(f" > {test}")

            if failed_tests:
                print("\nFailed tests: ")
                for test in failed_tests:
                    print(f" > {test}")
        else:
            print(
                " "*90, "All tests run successfully!\n"
            )

