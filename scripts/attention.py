import torch
from torch import nn


class Attention(nn.Module):
    def __init__(
        self, states=None, transform_states=False, state_dim=128, hidden_dim=128
    ):
        super().__init__()
        self.states = states
        self.transform_states = transform_states

        self.transformed_keys = None
        self.transformed_values = None
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim if states is None else states.shape[-1]

        self.attention_scores = None

        if transform_states:
            self.values_mlp = nn.Linear(
                in_features=state_dim, out_features=self.hidden_dim
            )
            self.keys_mlp = nn.Linear(
                in_features=state_dim, out_features=self.hidden_dim
            )

    def set_states(self, states):
        self.states = states
        return

    def get_alignment_vectors(self, query):
        alignment_vectors = torch.matmul(query, self.transformed_keys.transpose(-2, -1))
        alignment_vectors /= torch.sqrt(
            torch.as_tensor(self.transformed_keys.size()[-1:]).to(device=query.device)
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

        return torch.matmul(self.attention_scores, self.transformed_values)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        states=None,
        transform_states=False,
        narrow=False,
        hidden_dim=128,
        state_dim=128,
        num_heads=32,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim
        self.attention_scores = None
        self.narrow = narrow

        if isinstance(transform_states, list):
            if len(transform_states) < num_heads:
                transform_states += [False] * (num_heads - len(transform_states))
        else:
            transform_states = [transform_states] * num_heads

        self.transform_states = transform_states

        self.query_transform = nn.Linear(
            in_features=self.state_dim, out_features=self.hidden_dim
        )

        if not narrow:
            self.attention_heads = [
                Attention(
                    states=states,
                    transform_states=ts,
                    hidden_dim=hidden_dim,
                    state_dim=hidden_dim,
                )
                for ts in transform_states
            ]

            self.attention_heads = nn.ModuleList(self.attention_heads)

            self.context_transform = nn.Linear(
                in_features=sum([head.hidden_dim for head in self.attention_heads]),
                out_features=self.hidden_dim,
            )

        else:
            # self.query_transform = nn.LazyLinear(out_features=int(hidden_dim * 0.5))

            self.attention_heads = Attention(
                    states=states,
                    transform_states=True,
                    hidden_dim=int(hidden_dim/num_heads),
                    state_dim=int(hidden_dim/num_heads),
                )

            self.context_transform = nn.Linear(
                in_features=self.hidden_dim,
                out_features=self.hidden_dim,
            )

    def set_states(self, states):
        if self.narrow:
            self.attention_heads.set_states(
                self.get_dims_per_head(states)
            )
        else:
            for head in self.attention_heads:
                head.set_states(states)

        return

    def get_attention_scores(self):
        self.attention_scores = torch.stack(
            [head.attention_scores for head in self.attention_heads], dim=1
        )
        return

    def get_dims_per_head(self, x):
        N, L, H = x.shape
        x = x.view(N, L, self.num_heads, int(H/self.num_heads))
        return x.transpose(1, 2)

    def forward(self, query, mask=None):
        # query = self.query_transform(query) if self.narrow else query
        # context_vectors = [head(query, mask=mask) for head in self.attention_heads]

        if self.narrow:
            query = self.get_dims_per_head(
                self.query_transform(query)
            )

            N, n, L, h = query.shape
            context_vectors = self.attention_heads(query, mask=mask).view(N, L, n*h).contiguous()
        else:
            queries = [query] * self.num_heads

            context_vectors = [
                head(qry, mask=mask) for head, qry in zip(self.attention_heads, queries)
            ]

            context_vectors = torch.concat(context_vectors, dim=-1)

        return self.context_transform(context_vectors)
