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
        self,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        num_heads=32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.attention_scores = None
        self.narrow = narrow

        if isinstance(transform_states, list):
            if len(transform_states) < num_heads:
                transform_states += [False] * (num_heads - len(transform_states))
        else:
            transform_states = [transform_states] * num_heads

        self.attention_heads = [
            Attention(
                states=states,
                transform_states=ts,
                hidden_dim=int(hidden_dim * 0.5) if (self.narrow & ts) else hidden_dim,
            )
            for ts in transform_states
        ]

        self.transform_states = transform_states

        self.attention_heads = nn.ModuleList(self.attention_heads)

        self.context_transform = nn.LazyLinear(out_features=self.hidden_dim)

        if narrow:
            # self.query_transform = nn.LazyLinear(out_features=int(hidden_dim * 0.5))
            self.query_transforms = nn.ModuleList(
                [
                    nn.LazyLinear(
                        out_features=(int(hidden_dim * 0.5))
                        if (self.narrow & ts)
                        else hidden_dim
                    )
                    for ts in transform_states
                ]
            )

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
        # query = self.query_transform(query) if self.narrow else query
        # context_vectors = [head(query, mask=mask) for head in self.attention_heads]

        if self.narrow:
            queries = [
                (transform(query) if ts else query)
                for ts, transform in zip(self.transform_states, self.query_transforms)
            ]
        else:
            queries = [query] * self.num_heads

        print(queries[0].shape)

        context_vectors = [
            head(qry, mask=mask) for head, qry in zip(self.attention_heads, queries)
        ]

        context_vectors = torch.concat(context_vectors, dim=-1)

        return self.context_transform(context_vectors)
