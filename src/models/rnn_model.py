import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
            self, rnn_type, word_vocab_size, embedding_size,
            hidden_size, output_size, no_recurrent_layers,
            dropout_rate=0.5):
        super(RNNModel, self).__init__()
        self.ntoken = word_vocab_size
        self.drop = nn.Dropout(dropout_rate)
        self.encoder = nn.Embedding(word_vocab_size, embedding_size)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(
                nn, rnn_type)(
                embedding_size, hidden_size,
                no_recurrent_layers, dropout=dropout_rate, bidirectional=True)
        else:
            try:
                nonlinearity = {
                    'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for rnn_type was supplied, options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(
                embedding_size, hidden_size,
                no_recurrent_layers,
                nonlinearity=nonlinearity,
                dropout=dropout_rate, bidirectional=True)

        self.decoder = nn.Linear(2*hidden_size, output_size)
        self.init_weights()

        self.rnn_type = rnn_type
        self.hidden_size = hidden_size
        self.no_recurrent_layers = no_recurrent_layers

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input, lengths, hidden):
        emb = self.drop(self.encoder(input))
        packed_input = pack_padded_sequence(emb, lengths, batch_first=True)

        packed_output, (hidden_state, cell_state) = self.rnn(packed_input, hidden)

        output = torch.cat((cell_state[-1, :, :], cell_state[-2, :, :]), dim=-1)

        output = self.drop(output)
        decoded = self.decoder(output)

        return F.log_softmax(decoded, dim=1), (hidden_state, cell_state)

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(2*self.no_recurrent_layers, batch_size, self.hidden_size),
                    weight.new_zeros(2*self.no_recurrent_layers, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(
                2*self.no_recurrent_layers, batch_size, self.hidden_size)
