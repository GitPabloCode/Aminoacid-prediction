import torch
import math
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F 
import matplotlib.pyplot as plt
from torch.nn.modules.rnn import RNNBase, LSTMCell
from torch.nn import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class biLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size,  hidden_size):
        super(biLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first = True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        self.activation = nn.LogSoftmax(dim=2)


    def forward(self, input, avg_hidden = None, embedding_vector = None):
        embedded = self.embedding(input)
        output, _ = self.lstm(embedded)
        if avg_hidden is not None:
            mask = (input != 0).unsqueeze(2)
            masked_output = output * mask
            sum_embeddings = torch.sum(masked_output, dim=1)
            num_tokens = torch.sum(mask.squeeze(2), dim=1)
            avg_embeddings = sum_embeddings / num_tokens.unsqueeze(1)
            avg_hidden.append(avg_embeddings.detach().cpu()) 
        output = self.fc(output)
        output = self.activation(output)

        return output , avg_hidden
    

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size,  hidden_size):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=False, num_layers=2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.LogSoftmax(dim=2)


    def forward(self, input, avg_hidden = None):
        embedded = self.embedding(input)
        output, _ = self.lstm(embedded)
        output = self.fc(output)
        if(avg_hidden != None):
            mask = (input != 0).unsqueeze(2)
            masked_output = output * mask
            sum_embeddings = torch.sum(masked_output, dim=1)
            num_tokens = torch.sum(mask.squeeze(2), dim=1)
            avg_embeddings = sum_embeddings / num_tokens.unsqueeze(1)
            avg_hidden.append(avg_embeddings.detach().cpu()) 
        output = self.activation(output)
        return output, avg_hidden
    
    

'''
class mLSTM(RNNBase):
    def __init__(self, vocab_size, embedding_size, hidden_size, bias=True):
        super(mLSTM, self).__init__(
            mode='LSTM', input_size=vocab_size, hidden_size=hidden_size,
                 num_layers=1, bias=bias, batch_first=True,
                 dropout=0, bidirectional=True)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        w_im = torch.Tensor(hidden_size, embedding_size)
        w_hm = torch.Tensor(hidden_size, hidden_size)
        b_im = torch.Tensor(hidden_size)
        b_hm = torch.Tensor(hidden_size)
        self.w_im = Parameter(w_im)
        self.b_im = Parameter(b_im)
        self.w_hm = Parameter(w_hm)
        self.b_hm = Parameter(b_hm)
        

        self.lstm_cell = LSTMCell(embedding_size, hidden_size, bias)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, avg_hidden = None, hx=None):
        input = self.embedding(input)
        n_batch, n_seq, n_feat = input.size()

        if hx is None:
            hx = torch.zeros(n_batch, self.hidden_size).to(device)
            cx = torch.zeros(n_batch, self.hidden_size).to(device)
        else:
            hx, cx = hx

        steps = [cx.unsqueeze(1)]
        for seq in range(n_seq-1):
            mx = F.linear(input[:, seq, :], self.w_im, self.b_im) * F.linear(hx, self.w_hm, self.b_hm)
            hx = (mx, cx)
            hx, cx = self.lstm_cell(input[:, seq, :], hx)
            steps.append(cx.unsqueeze(1))

        output = torch.cat(steps, dim=1)
        output = self.linear(output)
        if(avg_hidden != None):
            mask = (input != 0).unsqueeze(2)
            masked_output = output * mask
            sum_embeddings = torch.sum(masked_output, dim=1)
            num_tokens = torch.sum(mask.squeeze(2), dim=1)
            avg_embeddings = sum_embeddings / num_tokens.unsqueeze(1)
            avg_hidden.append(avg_embeddings.detach().cpu())

        return output, avg_hidden

'''