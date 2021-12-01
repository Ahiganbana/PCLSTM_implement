import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random
import torch.nn.init as init

from torch.nn.modules import dropout, padding

torch.manual_seed(223)
random.seed(223)

class PCModule(nn.Module):

    def __init__(self, args):
        super(PCModule, self).__init__()
        self.args = args
        # 输入的特征维度
        Ci = args.input_dim
        Co = args.kernel_num
        padding_size = args.padding_size


        # CNN
        self.pconv = nn.Conv1d(in_channels = Ci,out_channels = Co, kernel_size = 3, stride=1, padding=padding_size)

        if self.args.cuda is True:
            self.pconv = self.pconv.cuda()
        
        self.batch_normalization_layer = nn.BatchNorm1d(Co)
        self.relu_activation_layer = nn.ReLU(inplace=True)
        self.time_pooling_layer = nn.AvgPool1d(kernel_size = 4, stride=2)
        self.dropout_layer = nn.Dropout(args.dropout)
    
    def forward(self, input):
        out = self.relu_activation_layer(self.pconv(input))
        out = self.time_pooling_layer(out)
        out = self.dropout_layer(out)
        return out


class LSTMModule(nn.Module):

    def __init__(self, args):
        super(LSTMModule, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        Li = args.input_dim

        self.lstm = nn.LSTM(Li, self.hidden_dim, num_layers = self.num_layers)

        if args.init_weight:
            print('Initinf W...')
            init.xavier_normal(self.lstm.all_weights[0][0], gain=np.sqrt(args.init_weight_value))
            init.xavier_normal(self.lstm.all_weights[0][1], gain=np.sqrt(args.init_weight_value))
        
        self.dropout = nn.Dropout(args.dropout)
        # self.full_connect = nn.Linear(14 * 2000, 6)
    
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = self.dropout(lstm_out)

        # b, _, _, = lstm_out.shape
        # out = lstm_out.view(b, -1)
        # out = self.full_connect(out)
        return lstm_out

class PCLSTMModule(nn.Module):

    def __init__(self, pcargs, lstmargs):
        super(PCLSTMModule, self).__init__()
        self.pcm1 = PCModule(pcargs[0])
        self.pcm2 = PCModule(pcargs[1])
        self.pcm3 = PCModule(pcargs[2])
        self.pcm4 = PCModule(pcargs[3])

        self.lstmm1 = LSTMModule(lstmargs[0])
        self.lstmm2 = LSTMModule(lstmargs[1])

        self.full_connect = nn.Linear(129 * 144, 6)
    
    def forward(self, input):
        pcm1_out = self.pcm1(input)
        pcm2_out = self.pcm2(pcm1_out)
        pcm3_out = self.pcm3(pcm2_out)
        pcm4_out = self.pcm4(pcm3_out)

        lstmm1_out = self.lstmm1(pcm4_out)
        lstmm2_out = self.lstmm2(lstmm1_out)

        b, _, _ = lstmm2_out.shape
        lstmm2_out = lstmm2_out.view(b, -1)

        out = self.full_connect(lstmm2_out)
        # out = lstmm2_out
        return out
