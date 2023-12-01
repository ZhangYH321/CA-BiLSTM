import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from my_dataset import MyDataSet,LSTM_MyDataSet
from lstm_model import LSTMModel
#from attn import LSTMModel
import math
from CNN_model import CNNModel
from utils import lstm_train_one_epoch,lstm_evaluate,get_split_data,get_eids,attn_train_one_epoch,attn_evaluate,load_npz

from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

setup_seed(2021)

region = "preMove_ALL"
save_names = ["CA-BiLSTM","MLP","CNN","LSTM"]

batch_size =128
num_classes = 2
epoches = 50
early_stop = 10




all_animals = ['SWC_054', 'ibl_witten_29', 'CSHL052','ZFM-02369', 'ZFM-01592', 'SWC_060', 'NYU-12', 'SWC_043', 'ibl_witten_27', 'CSHL058', 'CSHL059', 'ZM_2241', 'CSHL049', 'NYU-21', 'CSHL051', 'DY_018', 'NYU-45', 'NYU-48', 'ZFM-02373', 'DY_010', 'DY_009', 'ZFM-02370',
           'UCLA011', 'SWC_038', 'KS074', 'DY_020', 'NYU-37', 'KS044', 'NYU-29', 'NYU-47', 'SWC_042', 'SWC_058', 'ZM_3001', 'UCLA037', 'ibl_witten_25', 'ZFM-02372', 'KS023', 'ZFM-01936', 'CSH_ZAD_019', 'DY_016', 'UCLA034']


test_animal = ['ibl_witten_29', 'UCLA037', 'ZFM-02372', 'KS023', 'CSH_ZAD_019',"ZFM-02369","CSHL051","CSHL052","CSHL058","DY_018","NYU-37","NYU-45"]


def main(args):
    for save_name in save_names:
        print("现在运行的模型：", save_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        data_load_path = r"./segment_based"

        animals = []
        best_val = []
        best_loss = []
        #setup_seed(3407)
        for idx in np.arange(len(all_animals)):
            animal = all_animals[idx]
            if animal not in test_animal:
                continue
            print("--------------" * 2 + str(idx) + "----------------"*2)
            #attn_data = np.load(os.path.join(data_load_path, animal + "_cluster_data.npy"))
            all_data = load_npz(os.path.join(
                rf"{data_load_path}/segment_based_7",
                animal, animal + '_sameInterval.npz'))


            data, output, region, seq_len = all_data[0], all_data[1][:, 0], all_data[4], all_data[3]
            sampleNum, intervalNum, neuronNum = data.shape

            print(animal,neuronNum)


            idex = np.where(seq_len != 0)[0]
            data, output,  seq_len = data[idex], output[idex],  seq_len[idex]


            skf = StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
            # 做划分是需要同时传入数据集和标签
            dict_data = {}
            #记录每次交叉验证的正确率
            kf_best_acc = []
            kf_best_loss = []
            best_scores = []
            best_indexes = []
            print(data.shape[0],np.sum(output == 1) / len(output))


            # 设置随机数种子
            setup_seed(2021)
            for idx,(X_train_idx, X_val_idx) in enumerate(skf.split(data, output)):

                print(f"{animal}: 目前是第{idx}折")
                dict_data["train"] = [data[X_train_idx],output[X_train_idx],seq_len[X_train_idx]]
                dict_data["val"] = [data[X_val_idx],output[X_val_idx],seq_len[X_val_idx]]
                #加载数据
                train_data = dict_data["train"]
                val_data = dict_data["val"]

                train_input,train_output = train_data[0],train_data[1]
                val_input, val_output = val_data[0], val_data[1]


                model_name = "lstm"


                train_dataset = LSTM_MyDataSet(train_input,train_output,train_data[2])
                val_dataset = LSTM_MyDataSet(val_input, val_output,val_data[2])


                train_loader = torch.utils.data.DataLoader(train_dataset,
                                                           batch_size=batch_size,
                                                           #sampler=sampler,
                                                           shuffle=True,
                                                           num_workers=0,
                                                           collate_fn=train_dataset.collate_fn)

                val_loader = torch.utils.data.DataLoader(val_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=0,
                                                         collate_fn=val_dataset.collate_fn)


                if save_name == "CA-BiLSTM":
                    model = LSTMModel(input_size = train_input.shape[-1],hidden_layer_size = 128,num_layer = 2,bidirectional = True,output_size = 1,drop_rate=0.2,attn_name = save_name,max_seq= 20,device=device,IntervalNum=intervalNum).to(device)

                if save_name == "CNN":
                    model = CNNModel(input_size = train_input.shape[-1],channel_size = 128,output_size = 1,drop_rate = 0.2,device = device).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=0.05)

                lf = lambda x: ((1 + math.cos(x * math.pi / 50)) / 2) * (1 - args.lrf) + args.lrf  # cosine
                scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
                step = 0
                all_train_loss = []
                all_train_acc = []
                all_val_loss = []
                all_val_acc = []

                best_val_acc = 0
                best_val_loss = 1000 #最大值
                best_acc_loss = 0 #acc最大的时候的loss

                # 开始训练
                for epoch in range(epoches):
                    train_loss, train_acc = lstm_train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader,
                                                            device=device, epoch=epoch,model_name=model_name,attn_name = save_name,loss_type = "GCELoss",pre_method = "sameInterval")
                    scheduler.step()
                    current_val_loss, current_val_acc,score,index = lstm_evaluate(model=model, optimizer=optimizer, data_loader=val_loader, epoch=epoch,
                                                 device=device ,model_name=model_name,attn_name = save_name,loss_type = "GCELoss",pre_method = "sameInterval")


                    all_train_loss.append(train_loss)
                    all_train_acc.append(train_acc)
                    all_val_loss.append(current_val_loss)
                    all_val_acc.append(current_val_acc)


                    #loss值8轮不再下降就结束，最高正确率使用
                    if current_val_acc > best_val_acc:
                        best_val_acc = current_val_acc
                        best_acc_loss = current_val_loss
                        # torch.save(model.state_dict(),
                        #            "./lstm_weights/{}_{}_{}_model.pth".format(save_name, idx, animal))
                        best_score = score
                        correct_index = index

                    elif current_val_acc == best_val_acc and current_val_loss < best_acc_loss:
                        best_acc_loss = current_val_loss
                        # torch.save(model.state_dict(),
                        #            "./lstm_weights/{}_{}_{}_model.pth".format(save_name, idx, animal))
                        best_score = score
                        correct_index = index

                    if step > early_stop:
                        break

                    if best_val_loss >= current_val_loss:
                        best_val_loss = current_val_loss
                        step = 0
                    else:
                        step += 1

                #存acc最大的时候的分数和判断正确的下标
                best_scores.append(best_score)
                best_indexes.append(correct_index)

                kf_best_acc.append(best_val_acc)
                kf_best_loss.append(best_acc_loss)
                print(f"正确率：{best_val_acc}_{best_acc_loss}")

            #最后的显示
            animals.append(animal)
            print(animal,kf_best_acc,kf_best_loss)
            print("平均值：",np.array(kf_best_acc).mean())
            best_val.append(np.array(kf_best_acc))
            #animals.append(animal)
            best_loss.append(np.array(kf_best_loss))

        #显示下最高正确率和loss
        print(animals)
        print([val.mean() for val in best_val])
        print([loss.mean() for loss in best_loss])

        print(np.mean([val.mean() for val in best_val]))
        # 保存正确率
        # np.savez(f"attn_result/{save_name}.npz", animals,best_val,best_loss)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr',type=float,default=0.002)
    parser.add_argument('--lrf',type=float,default=0.01)

    args = parser.parse_args()
    main(args)