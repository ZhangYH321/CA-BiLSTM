import os

import numpy as np
import sys
sys.path.append('..')
from reproducible_ephys_functions import query   #, eid_list
import torch
from sklearn.metrics import f1_score
from tqdm import tqdm
from lstm_model import BCEFocalLoss,GCELoss
rng = np.random.default_rng(seed=10234567)

def get_eids():
    eids = ['56b57c38-2699-4091-90a8-aba35103155e',
            '72cb5550-43b4-4ef0-add5-e4adfdfb5e02',
            '746d1902-fa59-4cab-b0aa-013be36060d5',
            'dac3a4c1-b666-4de0-87e8-8c514483cacf',
            'a8a8af78-16de-4841-ab07-fde4b5281a03',
            'd2832a38-27f6-452d-91d6-af72d794136c',
            '6f09ba7e-e3ce-44b0-932b-c003fb44fb89',
            '73918ae1-e4fd-4c18-b132-00cb555b1ad2',
            'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
            'dda5fc59-f09a-4256-9fb5-66c67667a466',
            'ee40aece-cffd-4edb-a4b6-155f158c666a',
            '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
            '61e11a11-ab65-48fb-ae08-3cb80662e5d6',
            'ecb5520d-1358-434c-95ec-93687ecd1396',
            '54238fd6-d2d0-4408-b1a9-d19d24fd29ce',
            '51e53aff-1d5d-4182-a684-aba783d50ae5',
            'c51f34d8-42f6-4c9c-bb5b-669fd9c42cd9',
            '0802ced5-33a3-405e-8336-b65ebc5cb07c',
            'b03fbc44-3d8e-4a6c-8a50-5ea3498568e0',
            'db4df448-e449-4a6f-a0e7-288711e7a75a',
            '30c4e2ab-dffc-499d-aae4-e51d6b3218c2',
            '824cf03d-4012-4ab1-b499-c83a92c5589e',
            '41872d7f-75cb-4445-bb1a-132b354c44f0',
            'c7bf2d49-4937-4597-b307-9f39cb1c7b16',
            '4a45c8ba-db6f-4f11-9403-56e06a33dfa4',
            '7af49c00-63dd-4fed-b2e0-1b3bd945b20b',
            '781b35fd-e1f0-4d14-b2bb-95b7263082bb',
            '754b74d5-7a06-4004-ae0c-72a10b6ed2e6',
            '4b00df29-3769-43be-bb40-128b1cba6d35',
            'f140a2ec-fd49-4814-994a-fe3476f14e66',
            '862ade13-53cd-4221-a3fa-dda8643641f2',
            '3638d102-e8b6-4230-8742-e548cd87a949',
            'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',
            '2bdf206a-820f-402f-920a-9e86cd5388a4',
            'd9f0c293-df4c-410a-846d-842e47c6b502',
            '88224abb-5746-431f-9c17-17d7ef806e6a',
            'aad23144-0e52-4eac-80c5-c4ee2decb198',
            'd0ea3148-948d-4817-94f8-dcaf2342bbbe',
            '7f6b86f9-879a-4ea2-8531-294a221af5d0',
            'd23a44ef-1402-4ed7-97f5-47e9a7a504d9',
            '8a3a0197-b40a-449f-be55-c00b23253bbf',
            ]

    return eids


def get_traj(eids,one = None):
    traj = query(min_regions=0,one = None)
    tmp = []
    for eid in eids:
        for t in traj:
            if t['session']['id'] == eid:
                tmp.append(t)
                break
    traj = tmp

    return traj


import logging

def load_npz(filepath):
    # input is the mouse filepath
    # output is (spike, dirction, choice result, choice interval number, brain area)
    a = np.load(filepath, allow_pickle=True)
    return list(a[file] for file in a.files)

#logging.basicConfig(level=logging.INFO,filename="./log/demo1.log")
from sklearn.model_selection import train_test_split
def get_split_data(data,attn_data,output,seq_len):
    print("正负样本比例为：", len([i for i in output if i == 1]) / len(output))

    x = np.arange(data.shape[0])
    #output = [1 if o == 0 else 0 for o in output]
    # todo 我这里采用train_test_split来分配
    X_train_idx,X_val_idx,y_train,y_val = train_test_split(x,output,train_size=0.8,stratify=output)

    print("train的正负样本的比例：{} ，长度为 {}".format(len([i for i in y_train if i == 1]) / len(y_train),len(X_train_idx)))
    print("val的正负样本比例：{} ，长度为 {}".format(len([i for i in y_val if i == 1]) / len(y_val),len(y_val)))


    return {"train":[data[X_train_idx],y_train,seq_len[X_train_idx],attn_data[X_train_idx]],
            "val":[data[X_val_idx],y_val,seq_len[X_val_idx],attn_data[X_val_idx]],
            },X_train_idx,X_val_idx



def attn_train_one_epoch(model,optimizer,data_loader,device,epoch):
    model.train()
    #loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()

    sample_num = 0
    pred_list = []
    loss_function = torch.nn.BCELoss()

    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        seq,labels, seq_len = data
        optimizer.zero_grad()
        sample_num += seq.shape[0]
        y_pred = model(seq.to(device),seq_len).squeeze(-1)  # 压缩维度：得到输出，并将维度为1的去除

        single_loss = loss_function(y_pred, labels.to(device))
        accu_num += torch.eq(torch.round(y_pred), labels.to(device)).sum()
        pred_list.append(torch.round(y_pred))

        single_loss.backward()
        accu_loss += single_loss.detach()



        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)


        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def attn_evaluate(model,optimizer,data_loader,device,epoch,best_val_acc = 0):
    model.eval()
    #loss_function = torch.nn.CrossEntropyLoss()

    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)  # 累计预测正确的样本数
    pred_list = []
    target_list = []
    sample_num = 0
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    loss_function = torch.nn.BCELoss()

    with torch.no_grad():
        val_loader = tqdm(data_loader)

        for step, data in enumerate(val_loader):
            seq,labels,seq_len = data
            sample_num += seq.shape[0]

            y_pred = model(seq.to(device),seq_len).squeeze(-1)  # 压缩维度：得到输出，并将维度为1的去除
            single_loss = loss_function(y_pred, labels.to(device))
            accu_num += torch.eq(torch.round(y_pred), labels.to(device)).sum()

            pred_list.extend(torch.round(y_pred).cpu().numpy())
            target_list.extend(labels.numpy())

            accu_loss += single_loss

            val_loader.desc = "[val epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)

        f1 = f1_score(target_list,pred_list,average="binary")
        print("f1值为：{}".format(f1))



    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,best_val_acc

import torch.nn as nn
def lstm_train_one_epoch(model,optimizer,data_loader,device,epoch,model_name = "lstm",attn_name = "LSTM",loss_type = "BCELoss",pre_method = "sameInterval"):
    # 开始训练
    model.train()
    sample_num = 0
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    pred_list = []
    #loss_function = torch.nn.BCELoss()
    #triple_loss = MultiSimilarityLoss(margin = 10)

    if loss_type == "BCELoss":
        loss_function = torch.nn.BCELoss()
    elif loss_type == "GCELoss":
        loss_function = GCELoss()


    data_loader = tqdm(data_loader)
    for step, data in enumerate(data_loader):
        if model_name == "lstm":
            seq,labels,seq_len = data
            seq_len = None
        elif model_name == "attn_lstm":
            seq,attn_seq,labels,seq_len = data

        optimizer.zero_grad()
        sample_num += seq.shape[0]

        #todo LSTM可变长
        if pre_method == "sameTime":
            seq_len = data[2]
            sorted_id = sorted(range(len(seq_len)), key=lambda k: seq_len[k],reverse=True)
            seq_len = seq_len[sorted_id]
            seq = seq[sorted_id]
            labels = labels[sorted_id]
            #seq = nn.utils.rnn.pack_padded_sequence(seq,seq_len,batch_first=True)


        #attn+lstm
        if model_name == "attn_lstm":
            #attn_seq = attn_seq[sorted_id]
            y_pred,_ = model(seq.to(device),attn_seq.to(device),seq_len)  # 压缩维度：得到输出，并将维度为1的去除
            y_pred = y_pred.squeeze(-1)
        elif model_name == "lstm":
            y_pred,_ = model(seq.to(device), seq_len,attn_name = attn_name)  # 压缩维度：得到输出，并将维度为1的去除
            y_pred = y_pred.squeeze(-1)

        single_loss = loss_function(y_pred, labels.to(device))
        #添加一个三元组损失函数
        # triLoss = triple_loss(lstm_x,labels.to(device))
        # Loss = single_loss + triLoss

        accu_num += torch.eq(torch.round(y_pred), labels.to(device)).sum()

        pred_list.append(torch.round(y_pred))


        single_loss.backward()
        optimizer.step()
        accu_loss += single_loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                        accu_loss.item() / (step + 1),
                                                                        accu_num.item() / sample_num)


    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def lstm_evaluate(model,optimizer,data_loader,device,epoch,model_name = "lstm",attn_name = "LSTM",loss_type = "BCELoss",pre_method = "sameInterval"):

    model.eval()

    pred_list = []
    target_list = []
    sample_num = 0
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)

    TP = 0
    FP = 0
    FN = 0

    #loss_function = torch.nn.BCELoss()
    #loss_function = GCELoss()   # alph = u * max(1 , 1 + (M-Mk) / Mk)  u = 0.9
    #triple_loss = MultiSimilarityLoss(margin = 10)

    if loss_type == "BCELoss":
        loss_function = torch.nn.BCELoss()
    elif loss_type == "GCELoss":
        loss_function = GCELoss()

    with torch.no_grad():
        val_loader = tqdm(data_loader)

        for step, data in enumerate(val_loader):
            if model_name == "lstm":
                seq, labels, seq_len = data
            elif model_name == "attn_lstm":
                seq, attn_seq, labels, seq_len = data
            #loss_function = BCEFocalLoss(alpha= np.sum(labels.cpu().numpy() == 0) / np.sum(labels.cpu().numpy() == 1))
            sample_num += seq.shape[0]
            #todo 可变长
            if pre_method == "sameTime":
                sorted_id = sorted(range(len(seq_len)), key=lambda k: seq_len[k], reverse=True)
                seq_len = seq_len[sorted_id]
                seq = seq[sorted_id]
                labels = labels[sorted_id]
            else:
                seq_len = None


            #attn_seq = attn_seq[sorted_id]

            # attn+lstm
            if model_name == "attn_lstm":
                #attn_seq = attn_seq[sorted_id]
                y_pred,score = model(seq.to(device), attn_seq.to(device), seq_len)  # 压缩维度：得到输出，并将维度为1的去除
                score = score.cpu().numpy().squeeze()
                y_pred = y_pred.squeeze(-1)
                index = torch.eq(torch.round(y_pred), labels.to(device)).cpu().numpy()

            elif model_name == "lstm":
                y_pred,score = model(seq.to(device), seq_len,attn_name = attn_name)  # 压缩维度：得到输出，并将维度为1的去除
                y_pred = y_pred.squeeze(-1)
                #print(list(zip(labels,y_pred)))
                score = score.cpu().numpy().squeeze()
                #预测正确的和预测错误的索引
                index = torch.eq(torch.round(y_pred), labels.to(device)).cpu().numpy()

            single_loss = loss_function(y_pred, labels.to(device))
            # 添加一个三元组损失函数
            # triLoss = triple_loss(lstm_x, labels.to(device))
            # Loss = single_loss + triLoss

            accu_num += torch.eq(torch.round(y_pred), labels.to(device)).sum()

            pred_list.extend(torch.round(y_pred).cpu().numpy())
            target_list.extend(labels.numpy())

            accu_loss += single_loss


            val_loader.desc = "[val epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_num.item() / sample_num)



    return accu_loss.item() / (step + 1), accu_num.item() / sample_num,score,index

def bin_seq_spikes2D(spike_times, spike_clusters, cluster_ids, align_times, pre_time, post_time, bin_size,diff, weights=None):

    if np.floor(diff / bin_size).min() == 0:
        print("error")

    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    #得到trial的每个bin_size的时间点，要和firstMovement时刻时间点进行下比较
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    #真实的长度
    seq_len = np.array([1 if 0 == seq else seq for seq in np.floor(diff / bin_size)])

    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], cluster_ids.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        yscale, yind = np.unique(spike_clusters[ep[0]:ep[1]], return_inverse=True)
        nx, ny = [tscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx)) #计算了每个神经元在不同时间间隔的尖峰数
        r = np.bincount(ind2d, minlength=nx * ny, weights=w).reshape(ny, nx)
        bs_idxs = np.isin(cluster_ids, yscale)
        bins[i, bs_idxs, :int(seq_len[i])] = r[:, :int(seq_len[i])]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale,seq_len

def bin_seq_spikes_sameInterval(spike_times, spike_clusters, cluster_ids, align_times, interalNum,diff,linear_split = True, weights=None):
    #不同的反应时间，用相同的间隔，每个间隔时间不一致，具体是根据反应时间的大小
    tscale = np.zeros((len(diff),interalNum+1))
    for i,t in enumerate(diff):
        #线性分割
        if linear_split:
            tscale[i,:] = np.linspace(0,t,interalNum+1)
        #非线性分割
        else:
            interval = np.linspace(0,1,interalNum+1)
            tscale[i,:] = np.sqrt(interval) * t

    #每个trial的时间间隔 每个片段的实际时间
    ts = np.repeat(align_times[:, np.newaxis], tscale.shape[1], axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], cluster_ids.shape[0], interalNum))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        spike_time = spike_times[ep[0]:ep[1]]
        xind = np.zeros((len(spike_time)),dtype=int)         #索引
        for index,time in enumerate(spike_time):
            for k,interalTime in enumerate(ts[i][-2::-1]):
                if time > interalTime:
                    xind[index] = interalNum - k - 1
                    break
        w = weights[ep[0]:ep[1]] if weights is not None else None
        yscale, yind = np.unique(spike_clusters[ep[0]:ep[1]], return_inverse=True)
        nx, ny = tscale.shape[1], yscale.size # 5 48
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))  # 计算了每个神经元在不同时间间隔的尖峰数
        r = np.bincount(ind2d, minlength=nx * ny, weights=w).reshape(ny, nx)
        bs_idxs = np.isin(cluster_ids, yscale)
        bins[i, bs_idxs, :] = r[:, :-1]

    return bins



def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, diff, weights=None):

    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], n_bins))

    # 真实的长度
    seq_len = np.floor(diff / bin_size)

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        r = np.bincount(xind, minlength=tscale.shape[0], weights=w)
        bins[i, :int(seq_len[i])] = r[:int(seq_len[i])]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale

def bin_norm(times, events, pre_time, post_time, bin_size, diff,weights):
    bin_vals, t = bin_spikes(times, events, pre_time, post_time, bin_size,diff,weights=weights)
    bin_count, _ = bin_spikes(times, events, pre_time, post_time,bin_size,diff)
    bin_count[bin_count == 0] = 1
    bin_vals = bin_vals / bin_count

    return bin_vals, t

def preprocess_feature(feature):
    for i in range(feature.shape[-1]):
        min = feature[:,i].min()
        max = feature.max()
        feature[:,i] = (feature[:,i] - min) / (max - min)

    return feature


def cluster_preprocess_feature(feature_concat):
    xyz_offset = feature_concat.shape[-1]

    for i in range(xyz_offset):
        x_max = feature_concat[:, :, i].max()
        x_min = feature_concat[:, :, i].min()

        feature_concat[:, :, i] = 0.1 + 0.9 * (feature_concat[:, :, i] - x_min) / ((x_max - x_min) + 1e-9)

    return feature_concat
