import torch
import torch.nn as nn
import dataLoader.mydataset as mydataset
import numpy as np
import torch.nn.functional as F

class PanoramicModule(nn.Module):

    def __init__(self, inp, oup, padding):
        super(PanoramicModule, self).__init__()
        self.conv = nn.Conv1d(inp, oup, kernel_size=3, stride=1, padding=padding)
        self.bn = nn.BatchNorm1d(oup)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        out = self.relu(self.bn(self.conv(inputs)))
        out = self.pool(out)
        out = self.dropout(out)
        return out

class LSTMModule(nn.Module):

    def __init__(self):
        super(LSTMModule, self).__init__()
        self.lstm = nn.LSTM(144, 144, 2)
        self.dropout = nn.Dropout(0.75)
    
    def forward(self, input):
        lstm_out, _ = self.lstm(input)
        lstm_out = self.dropout(lstm_out)
        return lstm_out


class PCLSTMModule(nn.Module):

    def __init__(self):
        super(PCLSTMModule, self).__init__()
        self.pcm1 = PanoramicModule(14, 36, 1)
        self.pcm2 = PanoramicModule(36, 36, 1)
        self.pcm3 = PanoramicModule(36, 144, 1)
        self.pcm4 = PanoramicModule(144, 144, 7)

        self.lstmm1 = LSTMModule()
        self.lstmm2 = LSTMModule()

        self.full_connect = nn.Linear(129 * 144, 6)
    
    def forward(self, inputs):
        pcm1_out = self.pcm1(inputs)
        pcm2_out = self.pcm2(pcm1_out)
        pcm3_out = self.pcm3(pcm2_out)
        pcm4_out = self.pcm4(pcm3_out)

        pcm4_out = torch.transpose(pcm4_out, 1, 2).contiguous()
        print('pcm4_out', pcm4_out.shape)

        lstmm1_out = self.lstmm1(pcm4_out)
        print('lstmm1_out', lstmm1_out.shape)
        lstmm2_out = self.lstmm1(lstmm1_out)
        lstmm2_out = torch.transpose(lstmm2_out, 1, 2).contiguous()
        print('lstmm2_out', lstmm2_out.shape)

        b, _, _ = lstmm2_out.shape
        lstmm2_out = lstmm2_out.view(b, -1)

        out = self.full_connect(lstmm2_out)
        # out = lstmm2_out
        return out

fake_data = torch.rand((1280, 14, 2000))
print(type(fake_data))
model = PCLSTMModule()
out = model(fake_data)
print(out.shape)


# train_data = mydataset.Mydataset(r'D:\Learning\data\211128\normalization\train')
# fake_data = train_data.all_dataset
# fake_data_label = train_data.all_dataset_label

# tensor_data = torch.tensor(fake_data)
# tensor_data_label = torch.tensor(fake_data_label)
# print(tensor_data.shape)
# model = PCLSTMModule()
# out = model(tensor_data)
# print(out.shape)
# print(tensor_data.shape)
# print(tensor_data_label.shape)
# loss = F.cross_entropy(out, tensor_data_label.long())
# print(loss)