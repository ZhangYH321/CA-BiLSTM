import torch
from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self,data,output,seq_len = None,attn_seq = None):
        self.data = data
        self.output = output
        self.seq_len = None
        if seq_len is not None:
            self.seq_len = seq_len
        self.attn_seq = attn_seq

    def __len__(self):
        return len(self.output)

    def __getitem__(self, item):

        input = self.data[item]
        output = self.output[item]

        if self.seq_len is not None:
            return input,self.attn_seq[item],output,self.seq_len[item]
        return input,output


    @staticmethod
    def collate_fn(batch):
        #todo 可变长
        batch_input,attn_seq,batch_output,seq_len = zip(*batch)

        return torch.Tensor(batch_input),torch.Tensor(attn_seq),torch.Tensor(batch_output),torch.Tensor(seq_len)

class LSTM_MyDataSet(Dataset):
    def __init__(self,data,output,seq_len = None):
        self.data = data
        self.output = output
        self.seq_len = None
        if seq_len is not None:
            self.seq_len = seq_len

    def __len__(self):
        return len(self.output)

    def __getitem__(self, item):

        input = self.data[item]
        output = self.output[item]

        if self.seq_len is not None:
            return input,output,self.seq_len[item]
        return input,output


    @staticmethod
    def collate_fn(batch):
        #todo 可变长
        batch_input,batch_output,seq_len = zip(*batch)

        return torch.Tensor(batch_input),torch.Tensor(batch_output),torch.Tensor(seq_len)