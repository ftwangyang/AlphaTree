
from copy import copy
import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Dict, Optional


def initialize_weights_xavier(m, gain=1.0):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=gain)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def initialize_weights_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class DQNBase(nn.Module):
    def __init__(self, num_channels, embedding_dim=7*7*64):
        super(DQNBase, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            Flatten(),
        ).apply(initialize_weights_he)
        self.embedding_dim = embedding_dim

    def forward(self, states):
        batch_size = states.shape[0]
        state_embedding = self.net(states)
        assert state_embedding.shape == (batch_size, self.embedding_dim)
        return state_embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('_pe', pe)

    def forward(self, x):
        seq_len = x.size(0) if x.dim() == 2 else x.size(1)
        return x + self._pe[:seq_len]


class LSTMBase(nn.Module):
    def __init__(self, n_actions=50, embedding_dim=128, n_layers=2, dropout=0.1):
        super(LSTMBase, self).__init__()
        self._n_actions = n_actions
        self._token_emb = nn.Embedding(n_actions + 1, embedding_dim, 0)
        self._pos_enc = PositionalEncoding(embedding_dim)
        self._lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, obs):
        bs, seqlen = obs.shape
        beg = torch.full((bs, 1), fill_value=self._n_actions, dtype=torch.long, device=obs.device)
        obs = torch.cat((beg, obs.long()), dim=1)
        real_len = (obs != 0).sum(1).max()
        src = self._pos_enc(self._token_emb(obs))
        res = self._lstm(src[:, :real_len])[0]
        return res.mean(dim=1)


class TreeLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.W_leaf = nn.Linear(input_dim, 4 * hidden_dim)
        self.W_internal = nn.Linear(input_dim, 4 * hidden_dim)
        self.U_internal = nn.Linear(hidden_dim, 4 * hidden_dim, bias=False)
        self.W_forget = nn.Linear(input_dim, hidden_dim)
        self.U_forget = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                if 'W_leaf' in name or 'W_internal' in name:
                    param.data[self.hidden_dim:2*self.hidden_dim].fill_(1.0)

    def forward_leaf(self, x):
        gates = self.W_leaf(x)
        i, f, o, g = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = i * g
        h = o * torch.tanh(c)
        return h, c

    def forward_internal(self, x, children_h, children_c):
        h_sum = torch.stack(children_h, dim=0).sum(dim=0)
        gates = self.W_internal(x) + self.U_internal(h_sum)
        i_gate, _, o_gate, g_gate = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i_gate)
        o = torch.sigmoid(o_gate)
        g = torch.tanh(g_gate)
        c_from_children = torch.zeros_like(children_c[0])
        for child_h, child_c in zip(children_h, children_c):
            f_child = torch.sigmoid(self.W_forget(x) + self.U_forget(child_h))
            c_from_children = c_from_children + f_child * child_c
        c = i * g + c_from_children
        h = o * torch.tanh(c)
        return h, c


class TreeLSTMBase(nn.Module):
    def __init__(self, n_actions=50, embedding_dim=128, dropout=0.1):
        super(TreeLSTMBase, self).__init__()
        self._n_actions = n_actions
        self._embedding_dim = embedding_dim
        self._token_emb = nn.Embedding(
            num_embeddings=n_actions + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        self.tree_lstm_cell = TreeLSTMCell(
            input_dim=embedding_dim,
            hidden_dim=embedding_dim
        )
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.arity_map = {}
        self._build_arity_map()

    def _build_arity_map(self):
        try:
            from alphagen.config import OPERATORS
            from alphagen.rl.env.wrapper import (
                OFFSET_OP, OFFSET_FEATURE, OFFSET_DELTA_TIME,
                OFFSET_CONSTANT, OFFSET_SEP
            )
            self.arity_map[0] = -1
            for i, op in enumerate(OPERATORS):
                token_id = OFFSET_OP + i
                if hasattr(op, 'n_args'):
                    self.arity_map[token_id] = op.n_args
                elif hasattr(op, 'arity'):
                    self.arity_map[token_id] = op.arity
                else:
                    category = op.category_type() if hasattr(op, 'category_type') else None
                    if category in ['rolling', 'pair_rolling']:
                        self.arity_map[token_id] = 1
                    elif category in ['binary']:
                        self.arity_map[token_id] = 2
                    else:
                        self.arity_map[token_id] = 1
            for i in range(OFFSET_FEATURE, OFFSET_DELTA_TIME):
                self.arity_map[i] = 0
            for i in range(OFFSET_DELTA_TIME, OFFSET_CONSTANT):
                self.arity_map[i] = 0
            for i in range(OFFSET_CONSTANT, OFFSET_SEP):
                self.arity_map[i] = 0
            self.arity_map[OFFSET_SEP] = -1
            self.arity_map[self._n_actions] = -1
        except ImportError:
            for i in range(self._n_actions + 1):
                self.arity_map[i] = 0
            self.arity_map[0] = -1
            self.arity_map[self._n_actions] = -1

    def configure_arities(self, leaf_ids=None, unary_ids=None, binary_ids=None, special_ids=None):
        if leaf_ids:
            for tid in leaf_ids:
                self.arity_map[tid] = 0
        if unary_ids:
            for tid in unary_ids:
                self.arity_map[tid] = 1
        if binary_ids:
            for tid in binary_ids:
                self.arity_map[tid] = 2
        if special_ids:
            for tid in special_ids:
                self.arity_map[tid] = -1

    def get_arity(self, token_id):
        return self.arity_map.get(token_id, 0)

    def forward(self, obs):
        batch_size = obs.shape[0]
        device = obs.device
        root_embeddings = []
        for i in range(batch_size):
            seq = obs[i]
            seq = seq[seq != 0]
            if len(seq) == 0:
                root_h = torch.zeros(self._embedding_dim, device=device)
            else:
                root_h, _ = self._process_single_tree(seq)
            root_embeddings.append(root_h)
        result = torch.stack(root_embeddings, dim=0)
        result = self.layer_norm(result)
        result = self.dropout(result)
        return result

    def _process_single_tree(self, seq):
        device = seq.device
        stack = []
        for token_id in seq:
            token_id_int = token_id.item()
            arity = self.get_arity(token_id_int)
            if arity == -1:
                continue
            x = self._token_emb(token_id.unsqueeze(0)).squeeze(0)
            if arity == 0:
                h, c = self.tree_lstm_cell.forward_leaf(x)
            elif arity == 1:
                if len(stack) >= 1:
                    child_h, child_c = stack.pop()
                    h, c = self.tree_lstm_cell.forward_internal(x, [child_h], [child_c])
                else:
                    h, c = self.tree_lstm_cell.forward_leaf(x)
            elif arity == 2:
                if len(stack) >= 2:
                    right_h, right_c = stack.pop()
                    left_h, left_c = stack.pop()
                    h, c = self.tree_lstm_cell.forward_internal(x, [left_h, right_h], [left_c, right_c])
                elif len(stack) == 1:
                    child_h, child_c = stack.pop()
                    h, c = self.tree_lstm_cell.forward_internal(x, [child_h], [child_c])
                else:
                    h, c = self.tree_lstm_cell.forward_leaf(x)
            else:
                h, c = self.tree_lstm_cell.forward_leaf(x)
            stack.append((h, c))
        if len(stack) == 0:
            return (
                torch.zeros(self._embedding_dim, device=device),
                torch.zeros(self._embedding_dim, device=device)
            )
        return stack[-1]


class FractionProposalNetwork(nn.Module):
    def __init__(self, N=32, embedding_dim=7*7*64):
        super(FractionProposalNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, N)
        ).apply(lambda x: initialize_weights_xavier(x, gain=0.01))
        self.N = N
        self.embedding_dim = embedding_dim

    def forward(self, state_embeddings):
        batch_size = state_embeddings.shape[0]
        log_probs = F.log_softmax(self.net(state_embeddings), dim=1)
        probs = log_probs.exp()
        assert probs.shape == (batch_size, self.N)
        tau_0 = torch.zeros(
            (batch_size, 1), dtype=state_embeddings.dtype,
            device=state_embeddings.device)
        taus_1_N = torch.cumsum(probs, dim=1)
        taus = torch.cat((tau_0, taus_1_N), dim=1)
        assert taus.shape == (batch_size, self.N+1)
        tau_hats = (taus[:, :-1] + taus[:, 1:]).detach() / 2.
        assert tau_hats.shape == (batch_size, self.N)
        entropies = -(log_probs * probs).sum(dim=-1, keepdim=True)
        assert entropies.shape == (batch_size, 1)
        return taus, tau_hats, entropies


class CosineEmbeddingNetwork(nn.Module):
    def __init__(self, num_cosines=64, embedding_dim=7*7*64, noisy_net=False):
        super(CosineEmbeddingNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
        self.net = nn.Sequential(
            linear(num_cosines, embedding_dim),
            nn.ReLU()
        )
        self.num_cosines = num_cosines
        self.embedding_dim = embedding_dim

    def forward(self, taus):
        batch_size = taus.shape[0]
        N = taus.shape[1]
        i_pi = np.pi * torch.arange(
            start=1, end=self.num_cosines+1, dtype=taus.dtype,
            device=taus.device).view(1, 1, self.num_cosines)
        cosines = torch.cos(
            taus.view(batch_size, N, 1) * i_pi
        ).view(batch_size * N, self.num_cosines)
        tau_embeddings = self.net(cosines).view(
            batch_size, N, self.embedding_dim)
        return tau_embeddings


class QuantileNetwork(nn.Module):
    def __init__(self, num_actions, embedding_dim=128, dueling_net=False, noisy_net=False):
        super(QuantileNetwork, self).__init__()
        linear = NoisyLinear if noisy_net else nn.Linear
        if not dueling_net:
            self.net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, num_actions),
            )
        else:
            self.advantage_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, num_actions),
            )
            self.baseline_net = nn.Sequential(
                linear(embedding_dim, 64),
                nn.ReLU(),
                linear(64, 1),
            )
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

    def forward(self, state_embeddings, tau_embeddings):
        assert state_embeddings.shape[0] == tau_embeddings.shape[0]
        assert state_embeddings.shape[1] == tau_embeddings.shape[2]
        batch_size = state_embeddings.shape[0]
        N = tau_embeddings.shape[1]
        state_embeddings = state_embeddings.view(batch_size, 1, self.embedding_dim)
        embeddings = (state_embeddings * tau_embeddings).view(batch_size * N, self.embedding_dim)
        if not self.dueling_net:
            quantiles = self.net(embeddings)
        else:
            advantages = self.advantage_net(embeddings)
            baselines = self.baseline_net(embeddings)
            quantiles = baselines + advantages - advantages.mean(1, keepdim=True)
        return quantiles.view(batch_size, N, self.num_actions)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()
        self.mu_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma
        self.reset()
        self.sample()

    def reset(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def f(self, x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def sample(self):
        self.eps_p.copy_(self.f(self.eps_p))
        self.eps_q.copy_(self.f(self.eps_q))

    def forward(self, x):
        if self.training:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias
        return F.linear(x, weight, bias)

