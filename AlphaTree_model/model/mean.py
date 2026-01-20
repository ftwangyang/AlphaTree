

from torch import nn
import torch
from typing import Optional, List

from .base_model import BaseModel
from fqf_iqn_qrdqn1.network import NoisyLinear, LSTMBase, TreeLSTMBase


class MeanNetwork(BaseModel):

    def __init__(self,
                 num_actions,
                 embedding_dim=128,
                 dueling_net=False,
                 noisy_net=False,
                 use_tree_lstm=True):
        super(MeanNetwork, self).__init__()

        linear = NoisyLinear if noisy_net else nn.Linear

        self.use_tree_lstm = use_tree_lstm
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        self.dueling_net = dueling_net
        self.noisy_net = noisy_net

        if use_tree_lstm:
            self.dqn_net = TreeLSTMBase(
                n_actions=num_actions,
                embedding_dim=embedding_dim
            )
        else:
            self.dqn_net = LSTMBase(
                n_actions=num_actions,
                embedding_dim=embedding_dim
            )

        if not dueling_net:
            self.q_net = nn.Sequential(
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

    def configure_arities(self, leaf_ids=None, unary_ids=None, binary_ids=None, special_ids=None):
        if self.use_tree_lstm and hasattr(self.dqn_net, 'configure_arities'):
            self.dqn_net.configure_arities(
                leaf_ids=leaf_ids,
                unary_ids=unary_ids,
                binary_ids=binary_ids,
                special_ids=special_ids
            )

    def forward(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]

        if state_embeddings is None:
            state_embeddings = self.dqn_net(states)

        if not self.dueling_net:
            q_values = self.q_net(state_embeddings).view(batch_size, 1, self.num_actions)
        else:
            advantages = self.advantage_net(state_embeddings).view(batch_size, 1, self.num_actions)
            baselines = self.baseline_net(state_embeddings).view(batch_size, 1, 1)
            q_values = baselines + advantages - advantages.mean(dim=2, keepdim=True)

        return q_values

    def calculate_q(self, states=None, state_embeddings=None):
        assert states is not None or state_embeddings is not None
        batch_size = states.shape[0] if states is not None else state_embeddings.shape[0]

        q_values = self(states=states, state_embeddings=state_embeddings)
        q = q_values.squeeze(1)
        assert q.shape == (batch_size, self.num_actions)
        return q

