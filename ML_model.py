from data_utils import get_mouseFilePath, get_mouse, load_dataset,load_tensor_data
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import os
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn import preprocessing
from scipy.io import savemat
from glm import My_GLM
from utils import load_npz
from matplotlib import rcParams
import random

config = {
    "font.family": 'serif', # 衬线字体
    "font.size": 12, # 相当于小四大小
    "font.serif": ['SimSun'], # 宋体
    "mathtext.fontset": 'stix', # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    'axes.unicode_minus': False # 处理负号，即-号
}
rcParams.update(config)

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

setup_seed(1)

path_data = "segment_based"

def save_mat(mouseFilePath,encoding_input,y_mouse):

    animal = mouseFilePath.split("\\")[-2].split('/')[-1]
    savemat("spike_data/" + animal + ".mat", {"data": encoding_input, "label": y_mouse})

def models(config, test_animal,path_name,idx):
    mouse_acc = []
    mouse_std = []
    mouse_ratio = []
    mouse_num = []
    mouse_cluster_num = []
    mouse_names = []



    for mouse_name in test_animal:
        all_data = load_npz(os.path.join(
            rf"{path_name}\{path_data}_{idx}",
            mouse_name, mouse_name + '_sameInterval.npz'))
        data, output, regions, seq_len = all_data[0], all_data[1][:, 0], all_data[4], all_data[3]

        if mouse_name not in test_animal:
            continue

        mouse_names.append(mouse_name)
        origin_y = output
        origin_x = data

        #计算本身比例
        channel_ratio = np.sum(origin_y) / len(origin_y)
        mouse_ratio.append(channel_ratio)

        #print(mouse_name)
        x, y = load_dataset(origin_x.reshape((origin_x.shape[0], -1)), origin_y, add_interpcept=False)


        model_acc = {}
        model_var = {} #计算标准差
        for name, m in model.items():

            if name == "GLM":
                mean_scores = m.glm_model(x, y.reshape(-1))

            else:
                scores = cross_validate(m, x, y.reshape(-1), scoring=config["scoring"],
                                        cv=StratifiedKFold(config["cv"], shuffle=True, random_state=2020))
                mean_scores = np.mean(scores['test_accuracy'])
                var_scores = np.std(scores['test_accuracy'])

            model_acc[name] = mean_scores
            model_var[name] = var_scores



        mouse_acc.append(model_acc)
        mouse_std.append(model_var)

    return mouse_acc,mouse_ratio,mouse_num,mouse_cluster_num,mouse_names,mouse_std


if __name__ == '__main__':
    model = {
        'Logistic': LogisticRegression(penalty='l2', fit_intercept=False, solver='liblinear'),
        'DecisionTree': DecisionTreeClassifier(random_state=0),
        'SVM_Linear': SVC(kernel='linear'),
        'SVM_poly': SVC(kernel='poly'),
        'SVM_RBF': SVC(kernel='rbf'),
        "KNN":KNeighborsClassifier(),
        "GLM": My_GLM(C = 2),
    }
    # test_animal = ['ibl_witten_29', 'CSHL051', 'CSHL052', 'CSHL058', 'DY_018',
    #    'NYU-37', 'NYU-45', 'ZFM-02369', 'UCLA037', 'ZFM-02372', 'KS023',
    #    'CSH_ZAD_019','SWC_058']
    test_animal = [ "ibl_witten_29",]

    config = {
        'cv': 5,
        "scoring" :['accuracy'],
        'intervalNum': 4,
        'embedding_size': 15,
        "n_splits":5
    }


    diffInterval_acc = []
    path_name = rf"./{path_data}"
    for i in np.arange(1,11):
        mouse_acc,mouse_ratio,mouse_num,mouse_cluster_num,mouse_names,mouse_std = models(config, test_animal,path_name,i)
        diffInterval_acc.append(mouse_acc)

    #显示
    total_acc = []
    for i in np.arange(len(test_animal)):
        sameAnimal_acc = []
        for m_acc in diffInterval_acc:
            sameAnimal_acc.append(m_acc[i])
        total_acc.append(sameAnimal_acc)

    for j,m_acc in enumerate(total_acc):
        print(mouse_names[j])
        print(list(model.keys()))
        for i,acc in enumerate(m_acc):
            origin_acc = acc
            model_key = origin_acc.keys()
            origin_acc_value = [origin_acc[key] for key in model_key]
            method = '  \t\t'.join(model_key)
            origin_acc = '  \t'.join(str(format(n,'.4f')) for n in origin_acc_value)


            print(origin_acc)


    #画图
    total_acc = []
    for m_acc in diffInterval_acc:
        sameInterval_mean_acc = []
        for i, acc in enumerate(m_acc):
            sameInterval_mean_acc.append(list(acc.values()))
        mean_acc = np.array(sameInterval_mean_acc)
        total_acc.append(mean_acc)


    print(np.array(total_acc).mean(axis=1))

    print(total_acc)

    #np.savez("./ML_section_acc.npz",list(model.keys()),total_acc)

