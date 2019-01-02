'''
author: jingyuan
data: 2018/7/30
function: define model
'''
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
sys.path.insert(0, '../')


class LSTM(nn.Module):

    def __init__(self, args):
        super(LSTM, self).__init__()

        self.hidden_dim = 300
        self.num_layers = 1
        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        self.embed = nn.Embedding(V, D)
        # pretrained  embedding
        if args.word_Embedding:
            self.embed.weight.data.copy_(args.pretrained_weight)
        # lstm
        self.lstm = nn.LSTM(D, self.hidden_dim, dropout=0.4, num_layers=self.num_layers)
        # if args.init_weight:
        #     print("Initing W .......")
        #     # n = self.lstm.input_size * self.lstm
        #     init.xavier_normal(self.lstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
        #     init.xavier_normal(self.lstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))

        # linear
        self.hidden2label = nn.Linear(self.hidden_dim, C)
        # dropout
        self.dropout = nn.Dropout(0.4)
        self.dropout_embed = nn.Dropout(0.5)




    def forward(self, x):
        embed = self.embed(x)
        embed = self.dropout_embed(embed)
        x = embed.view(len(x), embed.size(1), -1)
        # lstm
        lstm_out, _ = self.lstm(x)
        # lstm_out, self.hidden = self.lstm(x, self.hidden)
        lstm_out = torch.transpose(lstm_out, 0, 1)
        lstm_out = torch.transpose(lstm_out, 1, 2)
        # pooling
        lstm_out = F.tanh(lstm_out)
        lstm_out = F.max_pool1d(lstm_out, lstm_out.size(2)).squeeze(2)
        lstm_out = F.tanh(lstm_out)
        # linear
        logit = self.hidden2label(lstm_out)
        return logit
