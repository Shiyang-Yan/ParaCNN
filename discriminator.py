

import torch
import torch.nn as nn

import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, linear_size, lin_dropout):
        super(Discriminator, self).__init__()

        self.hidden_size = hidden_size
        self.linear_size = linear_size


        self.rnn = nn.GRU(
            input_size, hidden_size,
            batch_first=True, bidirectional=True
        ).cuda()

        self.linears = nn.Sequential(
            nn.Linear(2*hidden_size, linear_size),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.Linear(linear_size, linear_size//2),
            nn.ReLU(),
            nn.Dropout(lin_dropout),
            nn.Linear(linear_size//2, 1)
        ).cuda()

    def forward(self, hidden_states):
        # hidden_states                                           # [batch_size * seq_len * hid_size]
        batch_size = hidden_states.size(0)
        initial_hidden = self.init_hidden(hidden_states.size(0))
        output, rnn_final_hidden = self.rnn(
            hidden_states, initial_hidden)                       # [2 * batch_size * hid_size]
        hidden_output = rnn_final_hidden
        hidden_output = hidden_output.view(batch_size, -1)
        scores = self.linears(hidden_output)                   # [batch_size * 1]
        #scores = F.sigmoid(scores)          # [batch_size * 1]
        return scores

    def init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, self.hidden_size).cuda()

        return hidden

