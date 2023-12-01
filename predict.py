import os
import json
from my_tsne import plot_embedding
import torch
from torchvision import transforms
from one.api import ONE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from my_dataset import MyDataSet,LSTM_MyDataSet
from reproducible_ephys_functions import save_data_path
from  utils import *
from lstm_model import LSTMModel,GCELoss
import numpy as np
import umap
import umap.plot
from sklearn.model_selection import KFold,StratifiedKFold
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
batch_size = 128

config = {
    "font.family": 'Times New Roman', # 衬线字体
    "font.size": 16, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)
all_animals = ['ibl_witten_29',"CSHL051","CSHL052","CSHL058","DY_018","NYU-37","NYU-45","ZFM-02369", 'UCLA037', 'ZFM-02372', 'KS023', 'CSH_ZAD_019']

test_animal = ["ibl_witten_29"]
#都选正确的trial
save_name = "CA_BiLSTM"


def plot_umap(animal,feature_x,y,save_name):


    y = pd.DataFrame(y)
    dic = {0: 'Left', 1: 'Right'}
    ls = []
    for index, value in y.iterrows():
        arr = np.array(value)[0]
        ls.append(dic[arr])

    embedding = umap.UMAP(random_state=50).fit_transform(feature_x)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x = embedding[:, 0], y = embedding[:, 1], hue=ls, style=ls,palette='Set1', sizes=10)
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f'{animal} {save_name} Data')
    # plt.xlabel('umap1')
    # plt.ylabel('umap2')
    #plt.legend(bbox_to_anchor=(0.35, 1.07),framealpha=0,ncol=2,fontsize=10)
    plt.legend(fontsize=16)
    plt.xticks([])
    plt.yticks([])
    #plt.savefig(f'./picture/UMAP/{animal}_{save_name}.png',bbox_inches = 'tight',dpi=1000)
    plt.show()




def plot_pred(val_idx=2):
    print("现在运行的模型：", save_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    for idx in np.arange(len(all_animals)):

        animal = all_animals[idx]
        if animal not in test_animal:
            continue
        print("--------------" * 2 + str(idx) + "----------------"*2)
        print(animal)
        all_data = load_npz(os.path.join(
            rf"./segment_based/segment_based_7",
            animal, animal + '_sameInterval.npz'))

        data, output, region, seq_len = all_data[0], all_data[1][:, 0], all_data[4], all_data[3]
        sampleNum, intervalNum, neuronNum = data.shape


        skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)

        for idx, (X_train_idx, X_val_idx) in enumerate(skf.split(data, output)):
            # 加载数据 验证数据
            if idx != val_idx:
                continue

            #plot_umap(animal,data[X_val_idx].reshape(data[X_val_idx].shape[0], -1), output[X_val_idx],"Raw")
            dataset = LSTM_MyDataSet(data[X_val_idx],output[X_val_idx],seq_len[X_val_idx])

            val_loader = torch.utils.data.DataLoader(dataset,
                                                     batch_size=batch_size,
                                                     shuffle=False,
                                                     num_workers=0,
                                                     collate_fn=dataset.collate_fn)


            model = LSTMModel(input_size=data.shape[-1], hidden_layer_size=128, num_layer=2, bidirectional=True,
                      output_size=1, drop_rate=0.2, attn_name=save_name, max_seq=20, device=device,IntervalNum=intervalNum).to(device)


            model_weight_path = f"./lstm_weights/{save_name}_{idx}_{animal}_model.pth"


            model.load_state_dict(torch.load(model_weight_path,map_location=device))
            model.eval()

            # y_pred, score, tt = model(torch.Tensor(data).to(device), attn_name=save_name)
            # plot_umap(animal, tt.cpu().detach().numpy(), output, "ME-BiLSTM")

            with torch.no_grad():
                val_loader = tqdm(val_loader)
                loss_function = GCELoss()
                pred_list = []
                target_list = []
                sample_num = 0
                accu_loss = torch.zeros(1).to(device)
                accu_num = torch.zeros(1).to(device)

                for step, data in enumerate(val_loader):

                    seq, labels, seq_len = data
                    sample_num += seq.shape[0]

                    y_pred, score,tt = model(seq.to(device), attn_name=save_name)  # 压缩维度：得到输出，并将维度为1的去除
                    y_pred = y_pred.squeeze(-1)
                    score = score.cpu().numpy().squeeze()


                    single_loss = loss_function(y_pred, labels.to(device))
                    accu_num += torch.eq(torch.round(y_pred), labels.to(device)).sum()

                    pred_list.extend(torch.round(y_pred).cpu().numpy())
                    target_list.extend(labels.numpy())

                    accu_loss += single_loss

                    #print(score)
                    # print(tt.cpu().numpy().shape)
                    # #T-SNE
                    # print('Starting compute t-SNE Embedding...')
                    # # print(label.shape)
                    # ts = TSNE(n_components=2, init='pca', random_state=0)
                    # # t-SNE降维
                    # reslut = ts.fit_transform(tt.cpu().numpy())
                    # # 调用函数，绘制图像
                    # fig = plot_embedding(reslut, output, 't-SNE Embedding of digits',accu_num.item() / sample_num)
                    # # 显示图像
                    # plt.show()
                    #UMAP
                    plot_umap(animal,tt.cpu().numpy(),output[X_val_idx],"CA-BiLSTM")
                    val_loader.desc = "[test] loss: {:.3f}, acc: {:.3f}".format(accu_loss.item() / (step + 1),
                                                                                      accu_num.item() / sample_num)

                print("预测正确率为：",accu_num.item() / sample_num)


if __name__ == '__main__':
    plot_pred()