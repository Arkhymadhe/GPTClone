import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, states=None, transform_states=False, hidden_dim=128):
        super().__init__()
        self.states = states
        self.transform_states = transform_states

        self.transformed_keys = None
        self.transformed_values = None
        self.hidden_dim = hidden_dim

        self.attention_scores = None

        if transform_states:
            self.values_mlp = nn.LazyLinear(out_features=self.hidden_dim)
            self.keys_mlp = nn.LazyLinear(out_features=self.hidden_dim)

    def set_states(self, states):
        self.states = states
        return

    def get_alignment_vectors(self, query):
        alignment_vectors = torch.bmm(query, self.transformed_keys.permute(0, 2, 1))
        alignment_vectors /= torch.sqrt(
            torch.as_tensor(self.transformed_keys.size()[-1:])
        )
        return alignment_vectors

    def forward(self, query, mask=None):
        self.transformed_keys = (
            self.keys_mlp(self.states) if self.transform_states else self.states
        )
        self.transformed_values = (
            self.values_mlp(self.states) if self.transform_states else self.states
        )

        alignment_vectors = self.get_alignment_vectors(query)

        if mask:
            alignment_vectors = mask * alignment_vectors

        self.attention_scores = torch.softmax(alignment_vectors, dim=-1).detach()

        return torch.bmm(self.attention_scores, self.transformed_values)


class MultiHeadAttention(nn.Module):
    def __init__(
        self, states=None, transform_states=False, hidden_dim=128, num_heads=32
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_scores = None

        if isinstance(transform_states, list):
            if len(transform_states) < num_heads:
                transform_states += [False] * (num_heads - len(transform_states))
        else:
            transform_states = [transform_states] * num_heads

        self.attention_heads = [
            Attention(
                states=states, transform_states=ts, hidden_dim=hidden_dim
            )
            for ts in transform_states
        ]

        self.attention_heads = nn.ModuleList(self.attention_heads)

        self.context_transform = nn.LazyLinear(out_features=self.hidden_dim)

    def set_states(self, states):
        for head in self.attention_heads:
            head.set_states(states)

        return

    def get_attention_scores(self):
        self.attention_scores = torch.stack(
            [head.attention_scores for head in self.attention_heads], dim=0
        )
        return

    def forward(self, query, mask=None):
        context_vectors = [head(query, mask=mask) for head in self.attention_heads]
        context_vectors = torch.concat(context_vectors, dim=-1)

        return self.context_transform(context_vectors)


if __name__ == "__main__":
    x = torch.randn(size=(32, 5, 128))
    query = torch.randn(size=(32, 10, 128))

    transform_states = [
        True, False, True, False,
    ]

    attn = MultiHeadAttention(num_heads=4, transform_states=transform_states, states=x)

    ans = attn(x)

    if hasattr(attn.attention_heads[0], "keys_mlp"):
        print(True)
    else:
        print(False)

    attn.get_attention_scores()

    print("Attention scores shape: ", attn.attention_scores.shape)
    print("Context vector shape: ", ans.shape, end="\n\n")

    for name, param in attn.named_parameters():
        print(name, " : ", param.shape)
