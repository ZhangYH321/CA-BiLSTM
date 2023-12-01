import torch
import torch.nn as nn
import numpy as np


class CNNModel(nn.Module):
    def __init__(self,input_size = 128,channel_size = 128,output_size = 1,num_layer = 2,bidirectional = True, drop_rate = 0.2,device = None):
        super(CNNModel, self).__init__()

        self.input_size = input_size
        self.emb_size = channel_size

        self.cnn_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=self.input_size,
                                    out_channels=200,
                                    kernel_size=h,
                                    ),
                          nn.BatchNorm1d(num_features=200),
                          nn.GELU(),
                          )
            # nn.MaxPool1d(kernel_size=self.input_size - h + 1))
            for h in [1,2,3,4]
        ])
        self.cnn_max_pool = nn.AdaptiveMaxPool2d((1,20))


        self.conv_linear = nn.Linear(in_features=20*4,out_features=self.emb_size * 2)

        self.device = device

        self.cat_linear = nn.Linear(in_features=self.emb_size * 2,
                                       out_features=self.emb_size)


        self.linear = nn.Linear(self.emb_size, output_size)
        self.drop = nn.Dropout(drop_rate)


    def get_cnn(self,output):

        #(batch_size,1,filterNum,4/3/2/1)
        batch_size = output.shape[0]
        output = output.transpose(1,2)
        out = [self.cnn_max_pool(conv(output).unsqueeze(dim=1).transpose(2,3)).squeeze(dim=1).squeeze(dim=1) for conv in self.cnn_convs]

        x = torch.cat(out,dim=-1).view(batch_size,-1)
        x = nn.LayerNorm(x.shape[-1],eps=1e-6).to(self.device)(x)

        return x



    def forward(self,x):

        cnn_x = self.get_cnn(x)
        x = self.conv_linear(cnn_x)

        x = self.act_layer(x)
        x = self.drop(x)
        x = self.sigmoid(self.linear(x))

        c_score = torch.ones((1, 4)).to(self.device)


        return x,c_score
