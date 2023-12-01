
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial
#padding 输入进来的是batch_size,seq_len


class LSTMModel(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)

    def __init__(self,input_size = 128,hidden_layer_size = 128,num_layer = 2,bidirectional = True,output_size = 1,drop_rate = 0.,attn_name = None,max_seq = 34,device = 'cuda:0',IntervalNum=7):

        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size,hidden_layer_size,num_layers=num_layer,batch_first=True,bidirectional=bidirectional,dropout = drop_rate)
        if bidirectional == True:
            self.hidden_layer_size = hidden_layer_size * 2
        else:
            self.hidden_layer_size = hidden_layer_size


        self.device = device
        self.act_layer = nn.ReLU()
        self.L_linear = nn.Linear(self.hidden_layer_size, self.hidden_layer_size)
        self.linear = nn.Linear(self.hidden_layer_size,output_size)
        self.drop = nn.Dropout(drop_rate)
        norm_layer = partial(nn.LayerNorm,eps=1e-6)
        #self.norm = norm_layer(input_size)
        self.norm1 = norm_layer(self.hidden_layer_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.GELU()


        if attn_name == "CA-BiLSTM":

            # 通道注意力机制
            in_planes = 128
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
            self.fc1 = nn.Linear(input_size, in_planes, bias=False).to(self.device)
            self.fc2 = nn.Linear(in_planes, input_size, bias=False).to(self.device)


    def get_in_channl_attn(self,output):
        #通道注意力机制
        output = output.transpose(1, 2)
        B = output.shape[0]
        avg_pool = self.avg_pool(output)
        max_pool = self.max_pool(output)
        avg_score = self.fc2(self.relu(self.fc1(avg_pool.view(B,-1))))
        max_score = self.fc2(self.relu(self.fc1(max_pool.view(B, -1))))
        score = self.tanh(avg_score+max_score).view(B, -1, 1)

        x = (output + output*(score.expand_as(output))).transpose(1,2)


        return x,score



    def forward(self,x,seq_len = None,attn_name = "LSTM"):

        #x维度为batch_size,seq_len,hidden
        B,_,C = x.shape

        #前置channel_attention
        if attn_name == "CA-BiLSTM":
            x,c_score = self.get_in_channl_attn(x)


        if seq_len == None:
            output, (h, c) = self.lstm(x)
            x = output[:, -1, :]

        else:
            rr = torch.zeros(size=(B, self.hidden_layer_size)).to(self.device)
            x = nn.utils.rnn.pack_padded_sequence(x, seq_len, batch_first=True)
            output, (h, c) = self.lstm(x)
            #x = output[:,-1,:]
            output, out_len = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
            for i,out in enumerate(output):
                rr[i] = out[out_len[i]-1,:]
            x = rr


        x = x.view(B, -1)
        x = self.norm1(x)
        #通道注意力
        x = self.L_linear(x)
        x = self.act_layer(x)
        tt = self.drop(x)
        x = self.sigmoid(self.linear(tt))

        return x,c_score

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.5, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        #pt = torch.sigmoid(predict) # sigmoide获取概率
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * target * torch.log(predict) - (1 - target) * torch.log(1 - predict)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class GCELoss(nn.Module):
    def __init__(self, num_classes=2, q=0.7):
        super(GCELoss, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, y_pred, labels):
        f = y_pred * labels + (1 - labels) * (1 - y_pred)
        loss = (1. - torch.pow(f, self.q)) / self.q
        return loss.mean()

if __name__ == "__main__":
    model = LSTMModel()
    x = model(torch.rand(4,30,1))
    print(x.shape)