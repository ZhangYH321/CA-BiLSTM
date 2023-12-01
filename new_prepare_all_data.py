import numpy as np
from utils import *
from one.api import ONE
from ibllib.atlas import AllenAtlas
import matplotlib.pyplot as plt
from brainbox.io.one import SpikeSortingLoader
from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS, repo_path, save_data_path
from reproducible_ephys_processing import bin_spikes, bin_spikes2D, compute_new_label,smoothing_kernel
import logging

import copy
import os
from collections import Counter

region = "preMove_ALL"

logging.basicConfig(level=logging.INFO,filename=f"./log/{region}_demo.log")

rng = np.random.default_rng(seed=10234567)

data_path = save_data_path(figure='figure11_my')

def prepare_data(one, eids= None,new_metrics=True,processType = "segment_based",interalNum = 5):
    brain_atlas = AllenAtlas()
    if eids is None:
        eids = get_eids()

    insertions = get_traj(eids)

    processType = "segment_based"  # "segment_based" or "time_interval"

    path = rf".\{processType}\{processType}_{interalNum}\\"

    all_data,all_output,seq_lens,eids,animals,regions = prepare_attn_data(one,insertions,brain_atlas,path,align_event = "stim_onset",
                                                                                           t_before = 0.5,t_after =  1,bin_size = 0.03,new_metrics=new_metrics,processType = processType,interalNum = interalNum)

    return all_data,all_output


def remap_choice_vals(choice):
    # raw choice vector has CW = 1 (correct response for stim on left),
    # CCW = -1 (correct response for stim on right) and viol = 0.  Let's
    # remap so that CW = 0, CCw = 1, and viol = -1
    choice_mapping = {1: 1, -1: 0, 0: -1}
    new_choice_vector = [choice_mapping[old_choice] for old_choice in choice]
    return new_choice_vector
def prepare_attn_data(one,insertions,brain_atlas,path,align_event = "movement_onset",t_before = 0.5,t_after =  1,bin_size = 0.03,new_metrics = True,processType = "segment_based",interalNum = 4):

    all_data = []
    all_output = []
    seq_lens = []
    eids = []

    animals = []
    regions = []
    session_ratios = []
    lessNum_animals = []
    good_animals = []
    zero_ratio = []



    for idx,trajectory in enumerate(insertions):

        eid = trajectory['session']['id']
        subject = trajectory['session']['subject']
        probe = trajectory['probe_name']
        print('processing {}: {}'.format(subject, eid))


        ba = brain_atlas or AllenAtlas()
        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
        clusters = sl.merge_clusters(spikes, clusters, channels)



        # Load in trials
        trials = one.load_object(eid, 'trials')
        if 'itiDuration' in trials.keys():
            del trials['itiDuration']
        trial_numbers = trials['stimOn_times'].shape[0]

        # CW -- 0   CCW -- 1
        y = np.expand_dims(remap_choice_vals(trials["choice"]), axis=1)

        if new_metrics:
            try:
                clusters['label'] = np.load(sl.files['clusters'][0].parent.joinpath('clusters.new_labels.npy'))
            except FileNotFoundError:
                new_labels = compute_new_label(spikes, clusters, save_path=sl.files['spikes'][0].parent)
                clusters['label'] = new_labels
                pass

        print("该神经元的长度为：{},trial为：{}".format(len(clusters["label"] == 1),trial_numbers))


        clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])


        cluster_idx = np.where(clusters['label'] == 1)[0]
        if cluster_idx.size == 0:
            continue
        cluster_id = clusters['cluster_id'][cluster_idx]
        # Find the index of spikes that belong to the chosen clusters
        spike_idx = np.isin(spikes['clusters'], cluster_id)

        assert np.all(cluster_id == cluster_idx)

        # 筛选完后的神经元的索引和脑区
        good_clusters = clusters["cluster_id"][cluster_idx]
        clusters_regions = clusters['rep_site_acronym'][cluster_idx]

        print("repeated site brain region counts: ", Counter(clusters['rep_site_acronym'][cluster_idx]))
        print("number of good clusters: ", cluster_id.shape[0])

        #todo 添加以前的选择信息,choice里面可能有nan， 注意判断一下
        select_idx = np.arange(len(trials["choice"]))


        # filter out trials with no choice
        choice_filter = np.where(trials['choice'] != 0)
        trials = {key: trials[key][choice_filter] for key in trials.keys()}

        # filter out trials with no contrast
        contrast_filter = ~np.logical_or(trials['contrastLeft'] == 0, trials['contrastRight'] == 0)

        trials = {key: trials[key][contrast_filter] for key in trials.keys()}

        nan_idx = np.c_[
            np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']), np.isnan(trials['goCue_times']),
            np.isnan(trials['response_times']), np.isnan(trials['feedback_times']), np.isnan(trials['stimOff_times'])]
        kept_idx = np.sum(nan_idx, axis=1) == 0

        trials = {key: trials[key][kept_idx] for key in trials.keys()}
        print("目前是筛选了所有的trail，不要对比度为0的，不要没有选择的，不要缺失时间点的，筛选完后的trial数为：",trials['stimOn_times'].shape[0])
        logging.info("目前是筛选了所有的trail，不要对比度为0的，不要没有选择的，不要缺失时间点的，筛选完后的trial数为：{}".format(trials['stimOn_times'].shape[0]))


        # select trials 刺激提示时间到移动轮子时间不超过0.4，且移动得到奖励时间到移动轮子时间小于0.9
        if align_event == 'movement_onset':
            ref_event = trials['firstMovement_times']
            diff1 = ref_event - trials['stimOn_times']
            diff2 = trials['feedback_times'] - ref_event
            t_select1 = np.logical_and(diff1 > 0.0, diff1 < t_before - 0.1)
            t_select2 = np.logical_and(diff2 > 0.0, diff2 < t_after - 0.1)
            t_select = np.logical_and(t_select1, t_select2)
        #todo select trials 这次我只要移动之前trial来判断 stim 0 - 1.5
        if align_event == "stim_onset":
            ref_event = trials['firstMovement_times']
            diff1 = ref_event - trials['stimOn_times']
            diff2 = trials['feedback_times'] - ref_event
            t_select1 = np.logical_and(diff1 > 0.03, diff1 < 1)
            t_select2 = np.logical_and(diff2 > 0.0, diff2 < 1)
            t_select = np.logical_and(t_select1, t_select2)

        #t_select = t_select1
        trials = {key: trials[key][t_select] for key in trials.keys()}



        select_idx = select_idx[choice_filter][contrast_filter][kept_idx][t_select]

        print("给点力啊，别太高了：", np.sum(trials["rewardVolume"] > 0) / len(select_idx))

        animal = one.eid2ref(eid)["subject"]
        animals.append(animal)
        animal_path = path + animal
        if not os.path.exists(animal_path):
            os.makedirs(animal_path)

        #输出为老鼠的选择
        output = trials["choice"]
        ref_event = trials['firstMovement_times']
        stim_event = trials["stimOn_times"]
        print("output的维度为：",output.shape)
        print("刺激提示时间到移动轮子时间不超过1s，且移动得到奖励时间到移动轮子时间小于0.9,筛选完后的trial数为：{}".format(trials['stimOn_times'].shape[0]))


        # Create expected output array （target）真实值
        if align_event == 'movement_onset':
            data, _ = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], good_clusters, ref_event,
                                 t_before, t_after, bin_size)
        elif align_event == "stim_onset":

            if processType == "segment_based":
                # time_interval_based:只取反应时间的内的spike数量，间隔时间为0.03s  总时间为 0-1s
                data, t,seq_len = bin_seq_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], good_clusters,
                                       trials['stimOn_times'],0, 1, interalNum * 0.01,diff=np.array(list(map(lambda x : x,diff1[t_select]))),
                                                   #weights = spikes["amps"][spike_idx]
                                    )

            if processType == "time_interval":
                # segment_based: 不同的反应时间，用相同的间隔，每个间隔时间不一致，具体是根据反应时间的大小
                data = bin_seq_spikes_sameInterval(spikes['times'][spike_idx], spikes['clusters'][spike_idx], good_clusters,
                                                trials['stimOn_times'], interalNum,diff=np.array(list(map(lambda x : x,diff1[t_select]))))
                seq_len = np.floor(diff1[t_select] / bin_size)

        #保存给ML
        np.savez(
            animal_path +'\\' + animal +
            '_sameInterval.npz',
            np.swapaxes(data,1,2), y[select_idx],trials["feedbackType"],seq_len,clusters_regions)


        data = np.swapaxes(data,1,2)


        print("输入的维度信息：",data.shape) #试验数，时间周期数，神经元个数
        logging.info("输入的维度信息：{}".format(data.shape))
        plus_minus_ratios = len([i for i in output if i == 1]) / len(output)
        print("正负样本比例为：", len([i for i in output if i == 1]) / len(output))
        session_ratios.append( len([i for i in output if i == 1]) / len(output))
        logging.info("正负样本比例为：{}".format(len([i for i in output if i == 1]) / len(output)))

        if len(trials["choice"]) * 0.2 < 50 or plus_minus_ratios > 0.55 or plus_minus_ratios < 0.45:
            lessNum_animals.append(animal)
        else:
            good_animals.append(animal)

        assert data.shape[0] == output.shape[0],print("output维度不一致")
        output = np.array([1 if i == 1 else 0 for i in output])
        #归一化
        all_data.append(data)
        all_output.append(output)
        eids.append(eid)
        seq_lens.append(seq_len)
        regions.append(clusters_regions)

        print("------------------------------------------"*2)
        logging.info("-------------------------------------------"*2)
    print(good_animals)
    print(zero_ratio)
    return all_data,all_output,seq_lens,eids,animals,regions


if __name__ == "__main__":
    one = ONE()
    prepare_data(one,eids=None)



