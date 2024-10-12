import os, time, copy

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import codecs, json
from sklearn import metrics
import shap

from multiprocessing import Process, Pool
from concurrent.futures import ProcessPoolExecutor
import logging, copy

from utils.model import AutoInt, HousePrice
from utils import utils
from utils import beeswarm

import cf_ml.cf_engine.engine as cf_engine
import cf_ml.dataset.dataset as cf_dataset

raw_cate_feat = ['school', 'grade', 'sex']
all_days = [11, 12, 13, 14]
all_clusters = list(range(7))
ckpt_name = 'norm_pretrain_epo_80_0.1494.ckpt'
global_step = 0
step_date = {}
step_setting = {}

'''init mongodb'''
from pymongo import MongoClient
client = MongoClient(host='127.0.0.1', port=27017)  # , socketTimeoutMS=60000, serverSelectionTimeoutMS=60000)
print(client.server_info())
global_coll = client['CHI2023']
print(list(global_coll.list_collections()))


def graph_metrics_process():
    with codecs.open('all_nodes.json', 'r') as f:
        raw_data = f.readlines()
        uid2id = json.loads(raw_data[0])
    id2uid = {val:key for key,val in uid2id.items()}

    metrics = ['cn', 'kcore', 'pr']
    days = 14
    days_list = []
    for d in range(days):
        print('processing day %d'%(d))
        df_list = []
        for met in metrics:
            list_path = os.path.join(utils.get_churn_d1(),'metrics/%s_list_%d.json' % (met, d))
            df = pd.read_json(list_path)
            df = df.T
            df.columns = [met]
            df_list.append(df)
        df_list = pd.concat(df_list, axis=1)
        df_list['date_group'] = 'Day%d' % (d+2)
        df_list['uid'] = [id2uid[int(id)] for id in df_list.index]
        days_list.append(df_list)
    days_list = pd.concat(days_list)
    days_list.to_csv(os.path.join(utils.get_churn_d1(), 'd1graph.csv'), index=False)
    print('processing done')

def update_portrait():
    d1label = pd.read_csv(os.path.join(utils.get_churn_d1(), 'xlx_d1label.txt'), sep='\t', engine='python')
    # d1label = d1label.drop(['deltatime_tomorrow'],axis=1)
    d1portrait = pd.read_csv(os.path.join(utils.get_churn_d1(), 'd1portrait.txt'), sep='\t', engine='python')
    d1portrait = d1portrait.drop(['t_when', 'onlinetime', 'totaltime'], axis=1)
    d1portrait_join = d1portrait.join(d1label.set_index(['uid', 'date_group']), on=['uid','date_group'], how='left', rsuffix='_1')
    d1portrait_join = d1portrait_join[~d1portrait_join['deltatime_tomorrow'].isna()]
    d1portrait_final = d1portrait_join.drop(['churn', 'deltatime', 'deltatime_tomorrow'], axis=1)
    d1portrait_final = d1portrait_final.rename(columns={'churn_1': 'churn', 'deltatime_1': 'deltatime'})
    d1portrait_final.to_csv(os.path.join(utils.get_churn_d1(), 'd1portrait_update.txt'), index=False)
    print(1)

def concat_all_feature():
    with codecs.open('all_nodes.json', 'r') as f:
        raw_data = f.readlines()
        uid2id = json.loads(raw_data[0])
    id2uid = {val:key for key,val in uid2id.items()}
    # features = ['portrait', 'graph', 'transition']
    feat_list = []
    path = os.path.join(utils.get_churn_d1(), 'd1portrait_update.txt')
    df = pd.read_csv(path, sep=' |\t|,', engine='python')
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    df = df.set_index(['uid', 'date_group'])
    # df['churn'] = [(delta==0)*3+((delta!=0)*churn) for churn, delta in zip(df['churn'], df['deltatime'])]
    feat_list.append(df)

    feat = 'graph'
    path = os.path.join(utils.get_churn_d1(), 'metrics.txt')
    df = pd.read_csv(path, sep=' |\t|,', engine='python')
    # df['uid'] = [np.int64(id2uid[id]) for id in df['id']]
    df = df.set_index(['uid', 'date_group'])
    feat_list.append(df)

    feat = 'transition'
    path = os.path.join(utils.get_churn_d1(), 'd1transition.txt')
    df = pd.read_csv(path, sep=' |\t|,', engine='python')
    df = df.reset_index()
    df = df.drop(['index'], axis=1)
    df = df.set_index(['uid', 'date_group'])
    # df = df.drop('churn', axis=1)
    feat_list.append(df)

    join_feat = feat_list[0].join(feat_list[1], how='left')
    join_feat = join_feat.join(feat_list[2], how='left')
    join_feat = join_feat.fillna(value=0)
    join_feat.loc[join_feat['churn'] == -1.0, ['churn']] = 3
    join_feat = join_feat.drop(join_feat.index[join_feat['churn']==-2.0], axis=0)

    join_feat.to_csv(os.path.join(utils.get_churn_d1(), 'concat_feat.csv'))
    print('cat features done')

def save_model(model, opt, name):
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': opt.state_dict()
    }, os.path.join(utils.get_churn_pretrain(), name))

def load_model(dataset,ckpt_name):
    model = AutoInt(dataset.data.shape[1], out_dim=4, feat_size=[len(dataset.cate_feat), len(dataset.rest_feat)],
                    voc_size=60, emb_dim=256, head_num=4, n_layers=4).to('cuda')
    ckpt = torch.load(os.path.join(utils.get_churn_pretrain(), ckpt_name))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    return model

def get_dataset(name):
    '''load feture'''
    testset = pd.read_csv(os.path.join(utils.get_churn_d1(), name), index_col=['uid', 'date_group'])
    '''test data'''
    test_label = testset['churn'].astype(np.int64)
    test_data = testset.drop('churn', axis=1)
    test_dataset = HousePrice(test_data, test_label, cate_feat=[0, 1, 4],
                              dtype=torch.long)  # ['school', 'grade', 'sex']
    return test_dataset

def print_metrics(prob_all, label_all):
    prob_idx = np.argmax(prob_all, axis=1)
    f1 = metrics.f1_score(label_all, prob_idx, average='weighted')
    auc = metrics.roc_auc_score(label_all, prob_all, multi_class='ovr')
    acc = metrics.accuracy_score(label_all, prob_idx)
    return f1, auc, acc

def focal_loss(y_pred, y_true, gamma=2, alpha=0.25, eps=1e-7):
    '''y_pred is after softmax'''
    pred_idx = torch.argmax(y_pred, dim=1)
    alpha_t = 1#(pred_idx == y_true)*alpha+(pred_idx != y_true)*(1-alpha)
    pt = -F.nll_loss(y_pred, y_true, reduction='none')+eps
    coeff = -(1-pt)**gamma
    loss = alpha_t * coeff * torch.log(pt)
    return loss.mean()

def topk_loss(y_pred, y_true):
    loss_func = nn.CrossEntropyLoss(reduction='none')
    k = int(y_pred.shape[0]*0.8)
    loss = loss_func(y_pred, y_true)
    loss = loss.topk(k)[0]
    return loss.mean()

def train(is_norm=0, loss_name='focal', ablation='null'):
    '''split train/test: deprecated'''
    # features = pd.read_csv(os.path.join(utils.get_churn_d1(), 'concat_feat.csv'), index_col=['uid', 'date_group'])
    # features = features.drop(['onlinetime', 'totaltime', 'class'], axis=1)
    # all_index = list(range(features.shape[0]))
    # np.random.shuffle(all_index)
    # split_idx = int(len(all_index)*0.8)
    # trainset = features.loc[features.index[all_index[:split_idx]]]
    # testset = features.loc[features.index[all_index[split_idx:]]]
    '''split train/test'''
    # features = pd.read_csv(os.path.join(utils.get_churn_d1(), 'concat_feat.csv'), index_col='date_group')
    # features = features.drop(['class'], axis=1)
    # trainset = features.loc[['Day%d'%i for i in range(2, 11)]]
    # testset = features.loc[['Day%d'%i for i in range(11, 15)]]
    # trainset = trainset.reset_index()
    # testset = testset.reset_index()
    # trainset.to_csv(os.path.join(utils.get_churn_d1(), 'trainset.csv'), index=False)
    # testset.to_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index=False)
    '''load csv'''
    trainset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'trainset.csv'), index_col=['uid', 'date_group'])
    testset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index_col=['uid', 'date_group'])
    description = pd.read_csv(os.path.join(utils.get_churn_d1(), 'description_full.csv'), index_col='name')
    '''feature ablation'''
    # ['school', 'grade', 'bindcash', 'combatscore', 'sex', 'deltatime',
    #  'churn', 'pr', 'kcore', 'cn', 'tran_school', 'tran_grade',
    #  'tran_deltatime', 'tran_bindcash', 'tran_combatscore', 'tran_sex']
    woportrait = ['churn', 'pr', 'kcore', 'cn', 'tran_school', 'tran_grade',
     'tran_deltatime', 'tran_bindcash', 'tran_combatscore', 'tran_sex']
    wosocial = ['school', 'grade', 'bindcash', 'combatscore', 'sex', 'deltatime',
     'churn', 'tran_school', 'tran_grade',
     'tran_deltatime', 'tran_bindcash', 'tran_combatscore', 'tran_sex']
    wotrans = ['school', 'grade', 'bindcash', 'combatscore', 'sex', 'deltatime',
     'churn', 'pr', 'kcore', 'cn']
    if ablation == 'woportrait':
        trainset = trainset[woportrait]
        testset = testset[woportrait]
    elif ablation == 'wosocial':
        trainset = trainset[wosocial]
        testset = testset[wosocial]
    elif ablation == 'wotrans':
        trainset = trainset[wotrans]
        testset = testset[wotrans]
    if is_norm:
        for col in trainset.columns:
            if col != 'churn':
                trainset[col] = (trainset[col].values - description.loc[col]['min'])/(description.loc[col]['max'] - description.loc[col]['min'])
                testset[col] = (testset[col].values - description.loc[col]['min']) / (description.loc[col]['max'] - description.loc[col]['min'])
    '''cate feat'''
    cate_feat = [list(trainset.columns).index(_feat) for _feat in raw_cate_feat]
    '''train data'''
    label = trainset['churn'].astype(np.int64)
    data = trainset.drop('churn', axis=1)
    dataset = HousePrice(data, label, cate_feat=cate_feat, dtype=torch.long)
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            collate_fn=dataset.collate_fn,
                            shuffle=True)
    '''test data'''
    test_label = testset['churn'].astype(np.int64)
    test_data = testset.drop('churn', axis=1)
    test_dataset = HousePrice(test_data, test_label, cate_feat=cate_feat, dtype=torch.long)#['school', 'grade', 'sex']
    test_loader = DataLoader(test_dataset,
                             batch_size=1024,
                             collate_fn=test_dataset.collate_fn,
                             shuffle=False)

    print('init dataset')

    in_dim = data.shape[1]
    epoch = 100
    model_type = 'local'#'deepctr'

    if model_type == 'local':
        model = AutoInt(in_dim, out_dim=4, feat_size=[len(dataset.cate_feat), len(dataset.rest_feat)],
                        voc_size=60, emb_dim=256, head_num=4, n_layers=4).to('cuda')

    # elif model_type == 'deepctr':
    #     model = AutoInt_dc(linear_feature_columns=list(range(0, 3)),
    #                        dnn_feature_columns=list(range(3, in_dim)),
    #                        device='cuda').to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=1e-3)
    # loss_name = 'focal'
    if loss_name == 'topk':
        loss = topk_loss
    elif loss_name == 'focal':
        loss = focal_loss
    else:
        loss = nn.CrossEntropyLoss()

    best_loss = 10
    for epo in range(epoch):
        prob_all = []
        label_all = []
        loss_list = []
        for _data, _label in dataloader:
            if model_type == 'local':
                emb, output = model(_data)
            # elif model_type == 'deepctr':
            #     cat_data = torch.cat(_data, dim=1)
            #     emb, output = model(cat_data)
            _loss = loss(output, _label)
            loss_list.append(_loss.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

            prob_all.append(output.detach().cpu().numpy())
            label_all.append(_label.detach().cpu().numpy())
        prob_all = np.concatenate(prob_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        f1, auc, acc = print_metrics(prob_all, label_all)
        print('avg loss: %f, f1: %.4f, auc: %.4f, acc: %.4f'%(np.mean(loss_list).item(), f1, auc, acc))
        if test_loader and epo % 10 == 0:
            model.eval()
            tmp_list = []
            prob_all = []
            label_all = []
            for _data, _label in test_loader:
                if model_type == 'local':
                    _, output = model(_data)
                # elif model_type == 'deepctr':
                #     cat_data = torch.cat(_data, dim=1)
                #     _, output = model(cat_data)
                _loss = loss(output, _label)
                tmp_list.append(_loss.clone().detach().cpu().numpy())
                prob_all.append(output.detach().cpu().numpy())
                label_all.append(_label.detach().cpu().numpy())
            mean_loss = np.mean(tmp_list).item()
            prob_all = np.concatenate(prob_all, axis=0)
            label_all = np.concatenate(label_all, axis=0)
            f1, auc, acc = print_metrics(prob_all, label_all)
            print('VALIDATION\navg loss: %f, f1: %.4f, auc: %.4f, acc: %.4f' % (mean_loss, f1, auc, acc))
            # if mean_loss < best_loss:
            print('epo: %d, loss: %f' % (epo, mean_loss))
            # best_loss = mean_loss
            save_name = '%s_pretrain_epo_%d_%.4f_%s.ckpt'
            if is_norm:
                save_name = 'norm_' + save_name
            save_model(model, optimizer, save_name % (loss_name, epo, mean_loss, ablation))
            model.train()


def inference(is_norm, ckpt_name, ablation='null'):
    '''load feture'''
    testset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index_col=['uid', 'date_group'])
    description = pd.read_csv(os.path.join(utils.get_churn_d1(), 'description_full.csv'), index_col='name')
    woportrait = ['churn', 'pr', 'kcore', 'cn', 'tran_school', 'tran_grade',
                  'tran_deltatime', 'tran_bindcash', 'tran_combatscore', 'tran_sex']
    wosocial = ['school', 'grade', 'bindcash', 'combatscore', 'sex', 'deltatime',
                'churn', 'tran_school', 'tran_grade',
                'tran_deltatime', 'tran_bindcash', 'tran_combatscore', 'tran_sex']
    wotrans = ['school', 'grade', 'bindcash', 'combatscore', 'sex', 'deltatime',
               'churn', 'pr', 'kcore', 'cn']
    if ablation == 'woportrait':
        testset = testset[woportrait]
    elif ablation == 'wosocial':
        testset = testset[wosocial]
    elif ablation == 'wotrans':
        testset = testset[wotrans]
    '''norm'''
    # is_norm = 1
    if is_norm:
        for col in testset.columns:
            if col != 'churn':
                testset[col] = (testset[col].values - description.loc[col]['min'])/(description.loc[col]['max'] - description.loc[col]['min'])
    '''cate feat'''
    cate_feat = [list(testset.columns).index(_feat) for _feat in raw_cate_feat]
    '''test data'''
    test_label = testset['churn'].astype(np.int64)
    test_data = testset.drop('churn', axis=1)
    test_dataset = HousePrice(test_data, test_label, cate_feat=cate_feat,
                              dtype=torch.long)  # ['school', 'grade', 'sex']
    test_loader = DataLoader(test_dataset,
                             batch_size=1024,
                             collate_fn=test_dataset.collate_fn,
                             shuffle=False)

    print('init dataset')

    in_dim = test_data.shape[1]
    model_type = 'local'  # 'deepctr'
    # ckpt_name = 'pretrain_epo_90_0.4385.ckpt'#'pretrain_epo_10_0.8241.ckpt'#'pretrain_epo_0_0.8120.ckpt'

    if model_type == 'local':
        # model = AutoInt(in_dim, out_dim=4, feat_size=[len(test_dataset.cate_feat), len(test_dataset.rest_feat)],
        #                 voc_size=60, emb_dim=64).to('cuda')
        model = AutoInt(in_dim, out_dim=4, feat_size=[len(test_dataset.cate_feat), len(test_dataset.rest_feat)],
                        voc_size=60, emb_dim=256, head_num=4, n_layers=4).to('cuda')
        ckpt = torch.load(os.path.join(utils.get_churn_pretrain(), ckpt_name))
        model.load_state_dict(ckpt['state_dict'])
        model = model.to('cuda')

    model.eval()
    tmp_list = []
    prob_all = []
    label_all = []
    for _data, _label in test_loader:
        if model_type == 'local':
            _, output = model(_data)
        prob_all.append(output.detach().cpu().numpy())
        label_all.append(_label.detach().cpu().numpy())
    prob_all = np.concatenate(prob_all, axis=0)
    label_all = np.concatenate(label_all, axis=0)
    f1, auc, acc = print_metrics(prob_all, label_all)
    print(
        'VALIDATION\nf1: %.4f, auc: %.4f, acc: %.4f' % (f1, auc, acc))

def check():
    trainset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'trainset.csv'), index_col=['uid', 'date_group'])
    testset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index_col=['uid', 'date_group'])
    remove_feat = ['tran_school', 'tran_grade', 'tran_deltatime', 'tran_bindcash', 'tran_combatscore', 'tran_sex']
    trainset1 = trainset.drop(remove_feat, axis=1)
    testset1 = testset.drop(remove_feat, axis=1)

    coll = []
    for trainrow in trainset1.iterrows():
        for testrow in testset1.iterrows():
            if (trainrow[1]==testrow[1]).all():
                coll.append([trainrow, testrow])

    print(1)


def makeDescription(filename, descript_name):
    testset = pd.read_csv(os.path.join(utils.get_churn_d1(), filename), index_col=['uid', 'date_group'])
    df = pd.DataFrame(columns=['name','type','decile','category','min'])
    prec0 = ['school', 'grade', 'bindcash', 'combatscore', 'sex', 'deltatime', 'churn', 'kcore']
    prec1 = list(set(list(range(len(testset.columns))))-set(prec0))
    for i, col in enumerate(testset.iteritems()):
        _prec = 1 if col[0] in prec0 else 5
        _item = {'name': col[0], 'type': 'numerical', 'decile': _prec, 'category': None, 'min': col[1].min(),
                 'max': col[1].max()}
        df = df.append(_item, ignore_index=True)
    df.to_csv(os.path.join(utils.get_churn_d1(), descript_name))
    print('generate description done')


def load_game_dataset():
    data_df = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'))
    description = pd.read_csv(os.path.join(utils.get_churn_d1(), 'description_full.csv'),
                              index_col='name').to_dict('index')
    for _, info in description.items():
        if type(info['category']) is str:
            info['category'] = info['category'].split(' ')
    return cf_dataset.Dataset('game', data_df, description, 'churn')

def load_game_dataset_mongo(day, setting):
    cond = {key: {'$gte':val[0], '$lte':val[1]} for key,val in setting.items()}
    cond['date_group'] = day
    cur_data = list(global_coll['current'].find(cond))
    # cur_data = list(global_coll['current'].find({}))
    if len(cur_data) == 0:
        return None
    data_df = pd.DataFrame(cur_data).drop(['churn', 'class', '_id'], axis=1)
    description = pd.read_csv(os.path.join(utils.get_churn_d1(), 'description_full.csv'),
                              index_col='name').to_dict('index')
    for _, info in description.items():
        if type(info['category']) is str:
            info['category'] = info['category'].split(' ')
    reorder_idx = [list(data_df.columns).index(_name) for _name in ['uid', 'date_group', 'school', 'grade', 'sex']]
    data_df = reorder(data_df, reorder_idx, order=0)# [0, 1, 2, 3, 6]
    reorder_idx = [list(data_df.columns).index('pred')]
    data_df = reorder(data_df, reorder_idx, order=1)
    '''change churn'''
    description['pred'] = description['churn']
    description.pop('churn')
    return cf_dataset.Dataset('game', data_df, description, 'pred')


def reorder(data, cate_feat, order=0):
    rest_feat = list(set(list(range(len(data.columns)))) - set(cate_feat))
    if order==0:
        df = data[data.columns[cate_feat+rest_feat]]
    else:
        df = data[data.columns[rest_feat+cate_feat]]
    return df


def counter_factual_example(setting, day, target):
    '''setting rephrase'''
    cf_range = copy.deepcopy(setting)
    changeable_attr = 'all'
    if 'change' in setting.keys():
        changeable_attr = setting['change']
        cf_range.pop('change')
    '''load dataset'''
    game_data = load_game_dataset_mongo(day, cf_range)
    if game_data is None:
        return None
    '''load model'''
    # testset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index_col=['uid', 'date_group'], nrows=1)
    testset = game_data.data.iloc[[0]].set_index(['uid','date_group'])
    test_label = testset['pred'].astype(np.int64)
    test_data = testset.drop('pred', axis=1)
    cate_feat = [list(test_data.columns).index(_feat) for _feat in raw_cate_feat]
    dataset = HousePrice(test_data, test_label, cate_feat=cate_feat,
                              dtype=torch.long)  # ['school', 'grade', 'sex']
    model = AutoInt(dataset.data.shape[1], out_dim=4, feat_size=[len(dataset.cate_feat), len(dataset.rest_feat)],
                    voc_size=60, emb_dim=256, head_num=4, n_layers=4, dataset=game_data, split=3, flag=1).to('cuda')
    ckpt = torch.load(os.path.join(utils.get_churn_pretrain(), ckpt_name))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    '''counterfactual init'''
    engine = cf_engine.CFEnginePytorch(game_data, model)
    '''generate counterfactual'''
    examples = game_data.data.drop(['uid','date_group', 'pred'], axis=1)# game_data.get_train_X(preprocess=False)
    # labels = np.zeros([examples.shape[0], 4])
    # eg_pred = game_data.data[['pred']].to_numpy().squeeze()
    # labels[np.arange(labels.shape[0]), eg_pred] = 1
    cf_setting = {'cf_range': cf_range, 'changeable_attr': changeable_attr, 'target': target}
    counterfactuals = engine.generate_counterfactual_examples(examples, setting=cf_setting)
    output = counterfactuals.valid
    return output


def save_pred():
    is_norm = 1
    '''load feture'''
    rawdata = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index_col=['uid', 'date_group'])
    testset = copy.deepcopy(rawdata)
    description = pd.read_csv(os.path.join(utils.get_churn_d1(), 'description_full.csv'), index_col='name')
    if is_norm:
        for col in testset.columns:
            if col != 'churn':
                testset[col] = (testset[col].values - description.loc[col]['min']) / (
                        description.loc[col]['max'] - description.loc[col]['min'])
    '''test data'''
    test_label = testset['churn'].astype(np.int64)
    test_data = testset.drop('churn', axis=1)
    cate_feat = [list(test_data.columns).index(_feat) for _feat in raw_cate_feat]
    test_dataset = HousePrice(test_data, test_label, cate_feat=cate_feat,
                              dtype=torch.long)  # ['school', 'grade', 'sex']
    test_loader = DataLoader(test_dataset,
                             batch_size=1024,
                             collate_fn=test_dataset.collate_fn,
                             shuffle=False)

    in_dim = test_data.shape[1]
    # ckpt_name = 'norm_pretrain_epo_50_0.0325.ckpt'  # 'pretrain_epo_10_0.8241.ckpt'#'pretrain_epo_0_0.8120.ckpt'

    model = AutoInt(in_dim, out_dim=4, feat_size=[len(test_dataset.cate_feat), len(test_dataset.rest_feat)],
                    voc_size=60, emb_dim=256, head_num=4, n_layers=4).to('cuda')
    ckpt = torch.load(os.path.join(utils.get_churn_pretrain(), ckpt_name))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')

    model.eval()
    prob_all = []
    for _data, _label in test_loader:
        _, output = model(_data)
        prob_all.append(output.detach().cpu().numpy())
    prob_all = np.concatenate(prob_all, axis=0)
    pred = np.argmax(prob_all, axis=1)

    rawdata = rawdata.reset_index()
    rawdata['pred'] = pred
    rawdata.to_csv(os.path.join(utils.get_churn_d1(), 'test_pred.csv'), index=False)


def chunk_insert(db, chunk, id):
    try:
        logging.info('insert %d' % id)
        res = db.insert_many(copy.deepcopy(chunk).to_dict(orient='record'))
    except Exception:
        raise Exception
    return res

def push_to_DB(filename, target_coll):
    coll = global_coll[target_coll]
    '''clean'''
    if coll.count_documents({}) != 0:
        coll.delete_many({})
    '''gridfs'''
    # import gridfs
    # fs = gridfs.GridFS(client.CHI2023)


    df_generator = pd.read_csv(os.path.join(utils.get_churn_d1(), filename), iterator=True, chunksize=10000)
    '''insert one'''
    # coll.insert_one(df_generator.__next__().to_dict())
    '''no multiprocess'''
    for chunk in df_generator:
        res = coll.insert_many(copy.deepcopy(chunk).to_dict(orient='record'))
    '''multiprocess.Pool'''
    # pool = Pool(8)
    # for i, chunk in enumerate(df_generator):
    #     pool.apply_async(chunk_insert, (coll, chunk, i,))
    # print('multiprocess start....')
    # pool.close()
    # pool.join()
    '''concurrent.future'''
    # with ProcessPoolExecutor(max_workers=8) as pool:
    #     future_list = []
    #     for i, chunk in enumerate(df_generator):
    #         future = pool.submit(chunk_insert, coll, chunk, i)
    #         future_list.append([i, future])
    #     for key, item in future_list:
    #         result = item.result()

    print('insert success')

def save_shap():
    is_norm = 1
    '''load feture'''
    rawdata = pd.read_csv(os.path.join(utils.get_churn_d1(), 'test_pred.csv'), index_col=['uid', 'date_group'])
    testset = copy.deepcopy(rawdata)
    testset = testset.drop(['pred'], axis=1)
    description = pd.read_csv(os.path.join(utils.get_churn_d1(), 'description_full.csv'), index_col='name')
    if is_norm:
        for col in testset.columns:
            if col not in ['churn', 'pred']:
                testset[col] = (testset[col].values - description.loc[col]['min']) / (
                        description.loc[col]['max'] - description.loc[col]['min'])
    '''test data'''
    test_label = testset['churn'].astype(np.int64)
    test_data = testset.drop('churn', axis=1)
    cate_feat = [list(test_data.columns).index(_feat) for _feat in raw_cate_feat]
    test_dataset = HousePrice(test_data, test_label, cate_feat=cate_feat,
                              dtype=torch.long)  # ['school', 'grade', 'sex']
    test_loader = DataLoader(test_dataset,
                             batch_size=16,
                             collate_fn=test_dataset.collate_fn,
                             shuffle=False)

    in_dim = test_data.shape[1]
    # ckpt_name = 'norm_pretrain_epo_50_0.0325.ckpt'  # 'pretrain_epo_10_0.8241.ckpt'#'pretrain_epo_0_0.8120.ckpt'

    model = AutoInt(in_dim, out_dim=4, feat_size=[len(test_dataset.cate_feat), len(test_dataset.rest_feat)],
                    voc_size=60, emb_dim=256, head_num=4, n_layers=4, split=3, flag=2).to('cuda')
    ckpt = torch.load(os.path.join(utils.get_churn_pretrain(), ckpt_name))
    model.load_state_dict(ckpt['state_dict'])
    model = model.to('cuda')
    model.eval()

    input_data = reorder(test_data, cate_feat).to_numpy()
    '''shap'''
    bg_index = np.random.choice(test_data.shape[0], 50, replace=False)
    bg_data = reorder(test_data.iloc[bg_index], cate_feat)
    background = bg_data.to_numpy()#torch.FloatTensor(bg_data.to_numpy()).to('cuda')
    test_summary = shap.kmeans(input_data, 50)
    explainer = shap.KernelExplainer(model, test_summary)
    shap_all = explainer.shap_values(input_data, nsamples=512)

    # shap_all = []
    # for _data, _label in test_loader:
    #     _data = torch.cat(_data, dim=1).cpu().numpy()
    #     _shap = explainer.shap_values(_data, nsamples=2*in_dim+1024)
    #     shap_all.append(_shap)
    # shap_all = np.concatenate(shap_all, axis=0)

    state = ['high', 'med', 'low', 'churn']
    for _stat, _shap in zip(state, shap_all):
        pd.DataFrame(_shap, columns=bg_data.columns).to_csv('shap_%s.csv' % _stat, index=False)

def update_shap():
    '''load test'''
    pred_df = pd.read_csv(os.path.join(utils.get_churn_d1(), 'test_pred.csv'))[['uid', 'date_group','pred']]
    '''update'''
    coll_list = ['high', 'med', 'low', 'churn']
    coll_list = [pd.read_csv(os.path.join(utils.get_churn_d1(), 'shap_%s.csv' % (item))) for item in coll_list]
    res_df = pd.DataFrame(np.zeros([pred_df.shape[0], pred_df.shape[1]+coll_list[0].shape[1]]), columns=list(pred_df.columns)+list(coll_list[0].columns))
    res_df[pred_df.columns] = pred_df
    for index, row in pred_df.iterrows():
        _state = int(row['pred'])
        res_df.loc[res_df.index[index], coll_list[0].columns] = coll_list[_state].iloc[index]
    if global_coll['state_shap'].count_documents({}) > 0:
        global_coll['state_shap'].delete_many({})
    global_coll['state_shap'].insert_many(res_df.to_dict(orient='record'))
    print('update shap success')

def add_class():
    '''local'''
    pred_df = pd.read_csv(os.path.join(utils.get_churn_d1(), 'test_pred.csv'))
    '''concat class'''
    features = pd.read_csv(os.path.join(utils.get_churn_d1(), 'concat_feat.csv'))
    pred_df = pred_df.join(features[['uid', 'date_group', 'class']].set_index(['uid', 'date_group']),
                           on=['uid', 'date_group'], how='left')
    coll = global_coll.test_pred

    if coll.count_documents({}) > 0:
        coll.delete_many({})

    res = coll.insert_many(pred_df.to_dict(orient='record'))

    print('add class success')

def api_init_current_mongo():
    global_coll['test_pred'].aggregate([
        {
            '$match': {}
        },
        {
            '$out': 'current'
        }
    ])

    global global_step
    global_step = 0
    step_date.clear()
    step_setting.clear()
    step_setting[0] = {str(i-11): {} for i in all_days}

    logging.info('clone test_pred to current')

def add_shap():
    '''get states'''
    churn_state = list(global_coll['test_pred'].aggregate([
        {
            '$project': {'uid':1, 'date_group':1, 'churn':1}
        }
    ]))
    '''search for shap'''
    coll_list = ['high', 'med', 'low', 'churn']
    coll_list = ['shap_%s'%(item) for item in coll_list]
    for row in churn_state:
        _state = int(row['pred'])
        _res = list(global_coll[coll_list[_state]].find_one({'uid': row['uid'], 'date_group':row['date_group']}))
        row.update(_res)
    '''insert'''
    global_coll['state_shap'].insert_many(churn_state)

    print('add shap success')

def api_timeline():
    pred_df = pd.read_csv(os.path.join(utils.get_churn_d1(), 'test_pred.csv'), index_col='date_group')
    ret = []
    days = ['Day%d'%i for i in all_days]
    for i, day in enumerate(days):
        _pred_day = pred_df.loc[[day]].reset_index()
        _data = {
            'id': i,
            'date': day,
            'high': _pred_day.loc[_pred_day['pred'] == 0].shape[0],
            'med': _pred_day.loc[_pred_day['pred'] == 1].shape[0],
            'low': _pred_day.loc[_pred_day['pred'] == 2].shape[0],
            'churn': _pred_day.loc[_pred_day['pred'] == 3].shape[0]
        }
        ret.append(_data)

    return ret


def api_timeline_mongo():
    current_data = list(global_coll['current'].find({}))
    pred_df = pd.DataFrame(current_data).set_index('date_group')
    ret = []
    days = ['Day%d' % i for i in all_days]
    for i, day in enumerate(days):
        if day in pred_df.index:
            _pred_day = pred_df.loc[[day]].reset_index()
            _data = {
                'id': i,
                'date': day,
                'high': _pred_day.loc[_pred_day['pred'] == 0].shape[0],
                'med': _pred_day.loc[_pred_day['pred'] == 1].shape[0],
                'low': _pred_day.loc[_pred_day['pred'] == 2].shape[0],
                'churn': _pred_day.loc[_pred_day['pred'] == 3].shape[0]
            }
        else:
            _data = {
                'id': i,
                'date': day,
                'high': 0,
                'med': 0,
                'low': 0,
                'churn': 0
            }

        ret.append(_data)

    return ret


def api_group(day):
    pred_df = pd.read_csv(os.path.join(utils.get_churn_d1(), 'test_pred.csv'))
    '''concat class'''
    features = pd.read_csv(os.path.join(utils.get_churn_d1(), 'concat_feat.csv'))
    pred_df = pred_df.join(features[['uid', 'date_group', 'class']].set_index(['uid', 'date_group']), on=['uid', 'date_group'], how='left')
    pred_df = pred_df.set_index('date_group')
    '''output'''
    ret = []
    for i in range(day[0], day[1]+1):
        day = 'Day%d' % all_days[i]
        _pred_day = pred_df.loc[[day]].reset_index()
        '''cluster'''
        _cluster_arr = []
        for c in all_clusters:
            _pred_day_c = _pred_day.loc[_pred_day['class'] == c]
            _clu = {
                'group': c,
                'num': [{'key': 'high', 'value': _pred_day_c.loc[_pred_day['pred'] == 0].shape[0]},
                        {'key': 'med', 'value': _pred_day_c.loc[_pred_day['pred'] == 1].shape[0]},
                        {'key': 'low', 'value': _pred_day_c.loc[_pred_day['pred'] == 2].shape[0]},
                        {'key': 'churn', 'value': _pred_day_c.loc[_pred_day['pred'] == 3].shape[0]}]
            }
            _cluster_arr.append(_clu)
        _data = {
            'id': i,
            'values': _cluster_arr,
            'pred': [{
                'id': i,
                'date': day,
                'high': _pred_day.loc[_pred_day['pred'] == 0].shape[0],
                'med': _pred_day.loc[_pred_day['pred'] == 1].shape[0],
                'low': _pred_day.loc[_pred_day['pred'] == 2].shape[0],
                'churn': _pred_day.loc[_pred_day['pred'] == 3].shape[0]
            }]
        }
        ret.append(_data)

    return ret


def api_group_mongo(day_num):
    day_num = [int(day_num[0]), int(day_num[1])]
    current_data = list(global_coll['current'].find({}))
    pred_df = pd.DataFrame(current_data)
    pred_df = pred_df.set_index('date_group')
    '''output'''
    ret = []
    for i in range(day_num[0], day_num[1]+1):
        day = 'Day%d' % all_days[i]
        _cluster_arr = []
        if day in pred_df.index:
            _pred_day = pred_df.loc[[day]].reset_index()
            '''cluster'''
            for c in all_clusters:
                _pred_day_c = _pred_day.loc[_pred_day['class'] == c]
                _clu = {
                    'group': c,
                    'num': [{'key': 'high', 'value': _pred_day_c.loc[_pred_day['pred'] == 0].shape[0]},
                            {'key': 'med', 'value': _pred_day_c.loc[_pred_day['pred'] == 1].shape[0]},
                            {'key': 'low', 'value': _pred_day_c.loc[_pred_day['pred'] == 2].shape[0]},
                            {'key': 'churn', 'value': _pred_day_c.loc[_pred_day['pred'] == 3].shape[0]}]
                }
                _cluster_arr.append(_clu)
            _data = {
                'id': i,
                'values': _cluster_arr,
                'pred': [{
                    'id': i,
                    'date': day,
                    'high': _pred_day.loc[_pred_day['pred'] == 0].shape[0],
                    'med': _pred_day.loc[_pred_day['pred'] == 1].shape[0],
                    'low': _pred_day.loc[_pred_day['pred'] == 2].shape[0],
                    'churn': _pred_day.loc[_pred_day['pred'] == 3].shape[0]
                }]
            }
        else:
            for c in all_clusters:
                _clu = {
                    'group': c,
                    'num': [{'key': 'high', 'value': 0},
                            {'key': 'med', 'value': 0},
                            {'key': 'low', 'value': 0},
                            {'key': 'churn', 'value': 0}]
                }
                _cluster_arr.append(_clu)
            _data = {
                'id': i,
                'values': _cluster_arr,
                'pred': [{
                    'id': i,
                    'date': day,
                    'high': 0,
                    'med': 0,
                    'low': 0,
                    'churn': 0
                }]
            }
        ret.append(_data)
    '''save setting'''
    global step_setting
    print('save setting at %d' % global_step)
    step_setting[global_step] = {str(d): {} for d in range(day_num[0], day_num[1]+1)}
    return ret


def api_next_mongo():
    '''next day'''
    current_data = list(global_coll['current'].find({}))
    cur_df = pd.DataFrame(current_data)
    all_uid = set(cur_df['uid'].to_list())
    last_date = sorted(list(set(cur_df['date_group'].to_list())), key=lambda x: int(x[3:]))[-1]
    last_date = int(last_date[3:])+1
    if last_date >all_days[-1]:
        return 'Err! There is no next day!'
    last_date = 'Day%d'%(last_date)
    print('getting next date of %s' % (last_date))
    next_df = pd.DataFrame(list(global_coll['test_pred'].find({'date_group': last_date, 'uid': {'$in': list(all_uid)}})))

    '''output'''
    ret = []
    _cluster_arr = []

    '''cluster'''
    for c in all_clusters:
        _pred_day_c = next_df.loc[next_df['class'] == c]
        _clu = {
            'group': c,
            'num': [{'key': 'high', 'value': _pred_day_c.loc[next_df['pred'] == 0].shape[0]},
                    {'key': 'med', 'value': _pred_day_c.loc[next_df['pred'] == 1].shape[0]},
                    {'key': 'low', 'value': _pred_day_c.loc[next_df['pred'] == 2].shape[0]},
                    {'key': 'churn', 'value': _pred_day_c.loc[next_df['pred'] == 3].shape[0]}]
        }
        _cluster_arr.append(_clu)
    did = int(last_date[3:])-11
    _data = {
        'id': did,
        'values': _cluster_arr,
        'pred': [{
            'id': did,
            'date': last_date,
            'high': next_df.loc[next_df['pred'] == 0].shape[0],
            'med': next_df.loc[next_df['pred'] == 1].shape[0],
            'low': next_df.loc[next_df['pred'] == 2].shape[0],
            'churn': next_df.loc[next_df['pred'] == 3].shape[0]
        }]
    }
    ret.append(_data)
    return ret


def api_link(day):
    '''local'''
    pred_df = pd.read_csv(os.path.join(utils.get_churn_d1(), 'test_pred.csv'))
    '''concat class'''
    features = pd.read_csv(os.path.join(utils.get_churn_d1(), 'concat_feat.csv'))
    pred_df = pred_df.join(features[['uid', 'date_group', 'class']].set_index(['uid', 'date_group']), on=['uid', 'date_group'], how='left')
    pred_df = pred_df.set_index('date_group')
    '''output'''
    ret = []
    fset, sset = None, None
    for i in range(day[0], day[1]):
        _data = []
        '''mongo'''
        # f_pred_day = pd.DataFrame(list(coll.find({'date_group': 'Day%d' % all_days[i]})), index='')
        '''local'''
        f_pred_day = pred_df.loc[['Day%d' % all_days[i]]].reset_index()
        s_pred_day = pred_df.loc[['Day%d' % all_days[i+1]]].reset_index()
        '''first day'''
        if fset is None:
            fset = {c: set(f_pred_day.loc[f_pred_day['class'] == c]['uid'].tolist()) for c in all_clusters}
        '''second day'''
        sset = {c: set(s_pred_day.loc[s_pred_day['class'] == c]['uid'].tolist()) for c in all_clusters}
        '''compute link'''
        for f_key, f_val in fset.items():
            _link = {
                'clu': f_key,
                'link': []
            }
            for s_key, s_val in sset.items():
                interset = f_val & s_val
                if len(interset) > 0:
                    _link['link'].append({
                        'tar': s_key,
                        'value': len(interset)
                    })
            if len(_link['link']):
                _data.append(_link)
        ret.append(_data)
        '''switch'''
        fset = sset
        sset = None

    return ret


def api_link_mongo(day):
    coll = global_coll['current']
    d0, d1 = int(day[0]), int(day[1])+1
    '''output'''
    ret = []
    fset, sset = None, None
    for i in range(d0, d1):
        _data = []
        '''mongo'''
        f_pred_day = pd.DataFrame(list(coll.find({'date_group': 'Day%d' % all_days[i]})))
        s_pred_day = pd.DataFrame(list(coll.find({'date_group': 'Day%d' % all_days[i+1]})))
        '''local'''
        # f_pred_day = pred_df.loc[['Day%d' % all_days[i]]].reset_index()
        # s_pred_day = pred_df.loc[['Day%d' % all_days[i+1]]].reset_index()
        '''first day'''
        if fset is None:
            fset = {c: set(f_pred_day.loc[f_pred_day['class'] == c]['uid'].tolist()) for c in all_clusters}
        '''second day'''
        sset = {c: set(s_pred_day.loc[s_pred_day['class'] == c]['uid'].tolist()) for c in all_clusters}
        '''compute link'''
        for f_key, f_val in fset.items():
            _link = {
                'clu': f_key,
                'link': []
            }
            for s_key, s_val in sset.items():
                interset = f_val & s_val
                if len(interset) > 0:
                    _link['link'].append({
                        'tar': s_key,
                        'value': len(interset)
                    })
            if len(_link['link']):
                _data.append(_link)
        ret.append(_data)
        '''switch'''
        fset = sset
        sset = None

    return ret


def api_shap(churn, is_plot=0):
    current = global_coll['current']
    '''get current and search'''
    cur_id = list(current.aggregate([
        {'$project': {'uid': 1, 'date_group':1,'_id':0}}
    ]))
    '''pandas manner'''
    cur_df = pd.DataFrame(cur_id)
    shap_df = pd.DataFrame(global_coll['state_shap'].find({}))
    # t1 = time.time()
    ret = cur_df.join(shap_df.set_index(['uid', 'date_group']), on=['uid', 'date_group'], how='left')
    # t2 = time.time()-t1
    # print('cur left join shap takes %f'%t2) #0.033869
    # t1 = time.time()
    # ret = shap_df.join(cur_df.set_index(['uid', 'date_group']), on=['uid', 'date_group'], how='right')
    # t2 = time.time() - t1
    # print('cur right join shap takes %f' % t2) #0.021821
    '''loop find'''
    # ret = [list(coll['state_shap'].find(_id)) for _id in cur_id]
    '''lookup according to current'''
    # ret = list(current.aggregate([
    #     {
    #         '$project': {'uid': 1, 'date_group':1}
    #     },
    #     {
    #         '$lookup': {
    #             'from': 'state_shap',
    #             'let': {'uid1': '$uid', 'date_group1': '$date_group'},
    #             'pipeline': [
    #                 {'$match': {'$expr': {
    #                     '$and': [
    #                         {'$eq': ['$uid', '$$uid1']},
    #                         {'$eq': ['$date_group', '$$date_group1']}
    #                     ]
    #                 }}}
    #             ],
    #             'as': 'join_feat'
    #         }
    #     },
    #     {
    #         '$unwind': '$join_feat'
    #     },
    #     {
    #         '$out': 'aggregate_result'
    #     }
    # ]))
    '''post-process'''
    ret = ret.drop(['uid', 'date_group', '_id'], axis=1)
    ret = ret.loc[ret['pred'] == churn]
    ret = ret.drop('pred', axis=1)
    if is_plot:
        # plot_res = shap.plots.beeswarm(ret.to_numpy())
        # plot_res = beeswarm.beeswarm(ret.to_numpy())
        # print('beeswarm plot')
        ret_np = ret.to_numpy()
        feat_idx = np.arange(ret_np.shape[0])[:, None].repeat(ret_np.shape[1], axis=1)
        ret = np.stack([feat_idx, ret_np]).transpose([1,2,0]).reshape([feat_idx.shape[0]*feat_idx.shape[1], 2])
        ret = ret.tolist()
    else:
        ret = ret.to_dict(orient='list')
    return ret

def api_boxplot_mongo():
    coll = global_coll['current']
    test_df = pd.DataFrame(list(coll.find({})))
    '''process'''
    test_df = test_df.drop(['churn', '_id', 'pred', 'class', 'uid', 'date_group'], axis=1)
    ret = []
    for feat in test_df.columns:
        _col = test_df[[feat]].to_numpy()
        _data = {
            'key': feat,
            'quartiles': [np.quantile(_col, 0.25).item(), np.quantile(_col, 0.5).item(), np.quantile(_col, 0.75).item()],
            'range': [_col.min().item(), _col.max().item()]
        }
        ret.append(_data)
    return ret


def api_boxplot():
    '''local'''
    test_df = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index_col=['date_group', 'uid'])
    '''process'''
    test_df = test_df.drop('churn', axis=1)
    ret = []
    for feat in test_df.columns:
        _col = test_df[[feat]].to_numpy()
        _data = {
            'key': feat,
            'quartiles': [np.quantile(_col, 0.25).item(), np.quantile(_col, 0.5).item(), np.quantile(_col, 0.75).item()],
            'range': [_col.min().item(), _col.max().item()]
        }
        ret.append(_data)
    return ret


def api_individual(args):
    uid, day = int(args['uid']), 'Day%d'%(args['day']+11)
    coll = global_coll['current']
    '''find'''
    res = coll.find_one({'date_group': day, 'uid': uid})
    ret = []
    if res is None:
        return ret
    for key, val in res.items():
        if key not in ['churn', '_id', 'pred', 'uid', 'date_group']:
            _data = {'uid': 0, 'key': key, 'value': val}
            ret.append(_data)
    return ret


def api_table(feat):
    coll = global_coll['test_pred']
    '''find'''
    cur_uid = None
    cur_df = None
    for day, feat_range in feat.items():
        day_cond = {'date_group': 'Day%d' % (int(day)+11)}
        for key, val in feat_range.items():
            day_cond[key] = {'$gte': val[0], '$lte': val[1]}
        day_res = pd.DataFrame(list(coll.find(day_cond)))
        if cur_df is None or len(cur_df) == 0:
            cur_df = day_res.set_index('uid')
            cur_uid = set(day_res['uid'].values.tolist())
        else:
            '''update uid'''
            day_uid = set(day_res['uid'].values.tolist())
            cur_uid = cur_uid & day_uid
            cur_df = cur_df.loc[list(cur_uid)]
            '''update df'''
            day_df = day_res.set_index('uid')
            day_df = day_df.loc[list(cur_uid)]
            cur_df = pd.concat([cur_df, day_df], axis=0)
    cur_df = cur_df.reset_index()
    # ret = cur_df.drop('_id', axis=1).to_dict(orient='record')
    '''update'''
    save_coll = global_coll['current']
    if save_coll.count_documents({}) > 0:
        save_coll.delete_many({})
    insert_df = copy.deepcopy(cur_df.to_dict(orient='record'))
    save_coll.insert_many(insert_df)

    cur_df['uid'] = cur_df['uid'].astype(str)
    ret = cur_df.drop('_id', axis=1).to_dict(orient='record')

    '''save setting'''
    global step_setting
    print('save setting at %d' % global_step)
    step_setting[global_step] = feat
    return ret


def api_cf(args):
    setting = args['setting']
    split_num = args['split_num']
    target = int(args['target'])
    days = ['Day%d'%(int(d)+11) for d in setting.keys()]
    '''real data'''
    raw = list(global_coll['current'].find({}))
    raw = pd.DataFrame(raw).set_index('date_group').drop(['churn', 'class', '_id', 'uid'], axis=1)
    '''counter factual data'''
    cf = []
    for _day, _set in setting.items():
        _day = 'Day%d' % (int(_day) + 11)
        print('producing CF of %s'%_day)
        _cf = counter_factual_example(_set, _day, target)
        if _cf is None:
            continue
        _cf.loc[:, ['date_group']] = _day
        cf.append(_cf)
    if len(cf) == 0:
        return 'ERROR: no data remains!'
    cf = pd.concat(cf, axis=0)
    cf = cf.set_index('date_group').drop('pred_pred', axis=1)
    print('CF ready')
    '''save data'''
    global step_setting
    print('save setting at %d'%global_step)
    step_setting[global_step] = setting
    if global_coll['cf_%d'%global_step].count_documents({})>0:
        global_coll['cf_%d' % global_step].delete_many({})
    if cf.shape[0]>0:
        _cf = cf.reset_index()
        global_coll['cf_%d'%global_step].insert_many(_cf.to_dict(orient='record'))
    '''special process'''
    # cf.loc[:, 'deltatime'] = (cf.loc[:, 'deltatime'] / 1000).astype(np.int)
    # cf.loc[:, 'tran_deltatime'] = (cf.loc[:, 'tran_deltatime'] / 1000).astype(np.int)
    # raw.loc[:, 'deltatime'] = (raw.loc[:, 'deltatime'] / 1000).astype(np.int)
    # raw.loc[:, 'tran_deltatime'] = (raw.loc[:, 'tran_deltatime'] / 1000).astype(np.int)
    '''formating'''
    ret = []
    states = ['high', 'med', 'low', 'churn']
    feat_columns = raw.drop(['pred'], axis=1).columns
    for _day in days:
        if _day in raw.index:
            draw = raw.loc[[_day]].reset_index()
        else:
            continue
        if _day in cf.index:
            dcf = cf.loc[[_day]].reset_index()
        else:
            dcf = None
        for _feat in feat_columns:
            draw_col = draw[_feat]
            real_value_range = [draw_col.min(), draw_col.max()]
            if dcf is not None:
                dcf_col = dcf[_feat]
                counter_value_range = [dcf_col.min(), dcf_col.max()]
                value_range = [min(real_value_range[0], counter_value_range[0]), max(real_value_range[1], counter_value_range[1])]
            else:
                dcf_col = None
                value_range = real_value_range
                counter_value_range = [0, 0]
            # draw_col, dcf_col = draw[_feat], dcf[_feat]
            # value_range = [min(draw_col.min(), dcf_col.min()), max(draw_col.max(), dcf_col.max())+1e-5]
            '''raw data'''
            if _feat == 'sex':
                raw_range = np.array([0, 1, 2, 3])
            elif value_range[1] > 1:
                # raw_range = np.ceil(np.arange(value_range[0], value_range[1] + 2e-5, (value_range[1] - value_range[0] + 1e-5) / split_num)).astype(np.int)
                raw_range = np.arange(value_range[0], value_range[1] + 2e-5, (value_range[1] - value_range[0] + 1e-5) / split_num)
                raw_range = np.concatenate([np.floor(raw_range[None, [0]]), np.ceil(raw_range[None, 1:])], axis=1).squeeze().astype(np.int)
                value_range = [raw_range[0].item(), raw_range[-1].item()]
            else:
                raw_range = np.arange(value_range[0], value_range[1]+2e-5, (value_range[1]-value_range[0]+1e-5) / split_num)
                value_range = [raw_range[0].item(), raw_range[-1].item()]
            raw_range = list(map(lambda x: x.item(), raw_range))

            # value_range = [raw_range[0], raw_range[-1]]
            # value_range = [min(draw_col.min(), dcf_col.min()), max(draw_col.max(), dcf_col.max())]
            raw_range_pair = list(zip(raw_range[:-1], raw_range[1:]))
            raw_range_data = [draw_col[(draw_col>=low) & (draw_col<up)] for low, up in raw_range_pair]
            count_range = [item.shape[0] for item in raw_range_data]
            count_range = [0, max(count_range)]
            values = []
            for _data, _range in zip(raw_range_data, raw_range_pair):
                if _data.shape[0] == 0:
                    continue
                _churn_data = [_data[draw['pred']==_c] for _c in range(4)]
                _data_segment = {
                    'name': _feat,
                    'range': _range,
                    'detail': [
                        {'key': _stat, 'value': _item.shape[0], 'count_range': count_range, 'value_range': value_range, 'split_num': split_num} for _item, _stat in zip(_churn_data, states)
                    ]
                }
                values.append(_data_segment)
            quartiles = [np.quantile(draw_col.values, 0.25).item(), np.quantile(draw_col.values, 0.5).item(), np.quantile(draw_col.values, 0.75).item()]
            '''cf data'''
            if dcf_col is not None:
                cf_range_data = [dcf_col[(dcf_col >= low) & (dcf_col < up)] for low, up in raw_range_pair]
                counter_count_range = [item.shape[0] for item in cf_range_data]
                counter_count_range = [0, max(counter_count_range)]
                counter_values = []
                for _data, _range in zip(cf_range_data, raw_range_pair):
                    if _data.shape[0] == 0:
                        continue
                    _churn_data = [_data[dcf['pred'] == _c] for _c in range(4)]
                    _data_segment = {
                        'name': _feat,
                        'range': _range,
                        'detail': [
                            {'key': _stat, 'value': _item.shape[0], 'count_range': counter_count_range, 'value_range': value_range,
                             'split_num': split_num} for _item, _stat in zip(_churn_data, states)
                        ]
                    }
                    counter_values.append(_data_segment)
                counter_quartiles = [np.quantile(dcf_col.values, 0.25).item(), np.quantile(dcf_col.values, 0.5).item(), np.quantile(dcf_col.values, 0.75).item()]
            else:
                counter_values = [0, 0]
                counter_count_range = [0, 0]
                counter_quartiles = [0, 0, 0]

            _ret = {
                'id': int(_day[3:])-11,
                'name': _feat,
                'values': values,
                'counter_values': counter_values,
                'value_range': value_range,
                'count_range': count_range,
                'counter_count_range': counter_count_range,
                'split_num': split_num,
                'quartiles': quartiles,
                'counter_quartiles': counter_quartiles,
                'real_value_range': real_value_range,
                'counter_value_range': counter_value_range,
                'band_domain': raw_range
            }
            ret.append(_ret)
            print('%s data done'%_day)

    return ret


def api_raw(args):
    setting = step_setting[global_step]
    split_num = args['split_num']
    days = ['Day%d'%(int(d)+11) for d in setting.keys()]
    '''real data'''
    raw = list(global_coll['current'].find({}))
    raw = pd.DataFrame(raw).set_index('date_group').drop(['churn', 'class', '_id', 'uid'], axis=1)
    '''formating'''
    ret = []
    states = ['high', 'med', 'low', 'churn']
    feat_columns = raw.drop(['pred'], axis=1).columns
    for _day in days:
        if _day in raw.index:
            draw = raw.loc[[_day]].reset_index()
        else:
            continue
        for _feat in feat_columns:
            draw_col = draw[_feat]
            real_value_range = [draw_col.min(), draw_col.max()]

            value_range = real_value_range
            '''raw data'''
            # if _feat == 'sex':
            #     raw_range = np.array([0, 1, 2, 3])
            # elif value_range[1] > 1:
            #     # raw_range = np.ceil(np.arange(value_range[0], value_range[1] + 2e-5, (value_range[1] - value_range[0] + 1e-5) / split_num)).astype(np.int)
            #     raw_range = np.arange(value_range[0], value_range[1] + 2e-5, (value_range[1] - value_range[0] + 1e-5) / split_num)
            #     raw_range = np.concatenate([np.floor(raw_range[None, [0]]), np.ceil(raw_range[None, 1:])], axis=1).squeeze().astype(np.int)
            # else:
            #     raw_range = np.arange(value_range[0], value_range[1]+2e-5, (value_range[1]-value_range[0]+1e-5) / split_num)
            # raw_range = list(map(lambda x: x.item(), raw_range))
            # value_range = [raw_range[0], raw_range[-1]]
            if _feat == 'sex':
                raw_range = np.array([0, 1, 2, 3])
            elif value_range[1] > 1:
                # raw_range = np.ceil(np.arange(value_range[0], value_range[1] + 2e-5, (value_range[1] - value_range[0] + 1e-5) / split_num)).astype(np.int)
                raw_range = np.arange(value_range[0], value_range[1] + 2e-5, (value_range[1] - value_range[0] + 1e-5) / split_num)
                raw_range = np.concatenate([np.floor(raw_range[None, [0]]), np.ceil(raw_range[None, 1:])], axis=1).squeeze().astype(np.int)
                value_range = [raw_range[0].item(), raw_range[-1].item()]
            else:
                raw_range = np.arange(value_range[0], value_range[1]+2e-5, (value_range[1]-value_range[0]+1e-5) / split_num)
                value_range = [raw_range[0].item(), raw_range[-1].item()]
            raw_range = list(map(lambda x: x.item(), raw_range))

            raw_range_pair = list(zip(raw_range[:-1], raw_range[1:]))
            raw_range_data = [draw_col[(draw_col>=low) & (draw_col<up)] for low, up in raw_range_pair]
            count_range = [item.shape[0] for item in raw_range_data]
            count_range = [0, max(count_range)]
            values = []
            for _data, _range in zip(raw_range_data, raw_range_pair):
                if _data.shape[0] == 0:
                    continue
                _churn_data = [_data[draw['pred']==_c] for _c in range(4)]
                _data_segment = {
                    'name': _feat,
                    'range': _range,
                    'detail': [
                        {'key': _stat, 'value': _item.shape[0], 'count_range': count_range, 'value_range': value_range, 'split_num': split_num} for _item, _stat in zip(_churn_data, states)
                    ]
                }
                values.append(_data_segment)
            quartiles = [np.quantile(draw_col.values, 0.25).item(), np.quantile(draw_col.values, 0.5).item(), np.quantile(draw_col.values, 0.75).item()]

            _ret = {
                'id': int(_day[3:])-11,
                'name': _feat,
                'values': values,
                'value_range': value_range,
                'count_range': count_range,
                'split_num': split_num,
                'quartiles': quartiles,
                'real_value_range': real_value_range,
                'band_domain': raw_range
            }
            ret.append(_ret)
            print('%s data done'%_day)

    return ret


def api_comparison():
    '''prepare data'''
    res_df = pd.DataFrame(list(global_coll['current'].find({})))
    days = list(set(list(res_df['date_group'].values)))
    res_df = res_df.drop(['uid', '_id', 'churn', 'pred', 'class'], axis=1).set_index('date_group')
    features = list(res_df.columns.values)
    '''formating'''
    ret = []
    for _day in days:
        for _feat in features:
            _col = res_df.loc[_day][[_feat]].to_numpy()
            _data = {
                'key': _feat,
                'id': int(_day[3:]),
                'quartiles': [np.quantile(_col, 0.25).item(), np.quantile(_col, 0.5).item(), np.quantile(_col, 0.75).item()],
                'range': [_col.min().item(), _col.max().item()]
            }
            ret.append(_data)
    return ret


def api_savestep(day):
    global global_step, step_date
    print('current step %d' % global_step)
    cur_step = copy.deepcopy(global_step)
    '''save current collection'''
    coll_name = 'step_%d' % cur_step
    if global_coll[coll_name].count_documents({})>0:
        global_coll[coll_name].delete_many({})
    global_coll['current'].aggregate([
        {
            '$match': {}
        },
        {
            '$out': coll_name
        }
    ])
    '''update step'''
    step_date[cur_step] = int(day)
    global_step += 1
    step_setting[global_step] = step_setting[cur_step]
    global_coll['cf_%d'%(cur_step)].aggregate([
        {
            '$match': {}
        },
        {
            '$out': 'cf_%d'%(global_step)
        }
    ])
    '''construct boxplot'''
    boxplot = {}
    for s in range(global_step):
        _data_df = pd.DataFrame(list(global_coll['step_%d'%s].find({})))
        _data_df = _data_df.drop(['_id', 'uid', 'class', 'churn', 'pred'], axis=1)
        d = step_date[s]
        if d != -1:
            dstr = 'Day%d' % (d+11)
            _data_df = _data_df.set_index('date_group').loc[dstr]
        else:
            _data_df = _data_df.drop('date_group', axis=1)
        for _feat in _data_df.columns:
            _col = _data_df[[_feat]].to_numpy()
            if _feat not in boxplot.keys():
                boxplot[_feat] = {
                    'key': _feat,
                    'values': [],
                    'counter_values': []
                }
            boxplot[_feat]['values'].append({
                'key': _feat,
                'id': s,
                'quartiles': [np.quantile(_col, 0.25).item(), np.quantile(_col, 0.5).item(), np.quantile(_col, 0.75).item()],
                'range': [_col.min().item(), _col.max().item()]
            })
        try:
            _cf_df = pd.DataFrame(list(global_coll['cf_%d' % s].find({})))
            _cf_df = _cf_df.drop(['_id', 'pred'], axis=1)
            if d != -1:
                dstr = 'Day%d' % (d + 11)
                _cf_df = _cf_df.set_index('date_group').loc[dstr]
                _setting = step_setting[s][str(day)]
                if 'change' in _setting:
                    _setting = _setting['change']
                else:
                    _setting = []
            else:
                _cf_df = _cf_df.drop('date_group', axis=1)
                _setting = set()
                for _item in step_setting[s].values():
                    if 'change' in _item.keys():
                        _setting |= list(_item['change'])
                _setting = list(_setting)
            for _feat in _data_df.columns:
                if _feat not in boxplot.keys():
                    boxplot[_feat] = {
                        'key': _feat,
                        'values': [],
                        'counter_values': []
                    }
                _cf_col = _cf_df[[_feat]].to_numpy()
                boxplot[_feat]['counter_values'].append({
                    'key': _feat,
                    'id': s,
                    'quartiles': [np.quantile(_cf_col, 0.25).item(), np.quantile(_cf_col, 0.5).item(),
                                  np.quantile(_cf_col, 0.75).item()],
                    'range': [_cf_col.min().item(), _cf_col.max().item()],
                    'change': int(_feat in _setting)
                })
        except Exception as exc:
            print('exc in save step: ', exc)
            for _feat in _data_df.columns:
                boxplot[_feat]['counter_values'] = []
        # if global_coll['cf_%d' % s].count_documents({}) > 0:
        #     _cf_df = pd.DataFrame(list(global_coll['cf_%d' % s].find({})))
        #     _cf_df = _cf_df.drop(['_id', 'churn'], axis=1)
        #     if d != -1:
        #         dstr = 'Day%d' % (d + 11)
        #         _cf_df = _cf_df.set_index('date_group').loc[dstr]
        #         _setting = step_setting[s][str(day)]
        #         if 'change' in _setting:
        #             _setting = _setting['change']
        #         else:
        #             _setting = []
        #     else:
        #         _cf_df = _cf_df.drop('date_group', axis=1)
        #         _setting = set()
        #         for _item in step_setting[s].values():
        #             if 'change' in _item.keys():
        #                 _setting |= list(_item['change'])
        #         _setting = list(_setting)
        #     for _feat in _data_df.columns:
        #         if _feat not in boxplot.keys():
        #             boxplot[_feat] = {
        #                 'key': _feat,
        #                 'values': [],
        #                 'counter_values': []
        #             }
        #         _cf_col = _cf_df[[_feat]].to_numpy()
        #         boxplot[_feat]['counter_values'].append({
        #             'key': _feat,
        #             'id': s,
        #             'quartiles': [np.quantile(_cf_col, 0.25).item(), np.quantile(_cf_col, 0.5).item(),
        #                           np.quantile(_cf_col, 0.75).item()],
        #             'range': [_cf_col.min().item(), _cf_col.max().item()],
        #             'change': int(_feat in _setting)
        #         })
        # else:
        #     for _feat in _data_df.columns:
        #         boxplot[_feat]['counter_values'] = []
    boxplot = list(boxplot.values())
    '''construct churnrate'''
    churnrate = []
    vitality = ['high', 'med', 'low', 'churn']
    for s in range(global_step):
        _data_df = pd.DataFrame(list(global_coll['step_%d'%s].find({})))
        _data_df = _data_df[['date_group', 'pred']]
        _days = _data_df['date_group'].value_counts().sort_index()
        _data_df = _data_df.set_index('date_group')
        '''each day'''
        values = []
        for key, val in _days.items():
            d = int(key[3:]) - 11
            _date = _data_df.loc[key].to_numpy()
            status = [{'name': vitality[c], 'value': (_date==c).sum().item()} for c in range(4)]
            values.append({
                'step': s,
                'date': d,
                'churn_rate': status[-1]['value']/len(_date),
                'day_len': len(_days),
                'num': len(_date),
                'status': status
            })
        avg_churn = np.average([item['churn_rate'] for item in values]).item()

        churnrate.append({
            'step': s,
            'metric': avg_churn,
            'values': values
        })

    ret = {
        'step': cur_step,
        'boxplot': boxplot,
        'churnrate': churnrate
    }
    return ret


def api_getstep(cur_step):
    '''save current collection'''
    coll_name = 'step_%d' % cur_step
    global_coll[coll_name].aggregate([
        {
            '$match': {}
        },
        {
            '$out': 'current'
        }
    ])
    data_df = pd.DataFrame(list(global_coll['current'].find({})))
    data_df = data_df.drop(['_id', 'uid', 'class', 'churn', 'pred'], axis=1)
    '''construct boxplot'''
    ret = {}
    days = data_df['date_group'].value_counts()
    data_df = data_df.set_index('date_group')
    for key, val in days.items():
        d = int(key[3:]) - 11
        _date = data_df.loc[key]
        ret[d] = {_feat: [_col.min(), _col.max()] for _feat, _col in _date.iteritems()}
        if 'change' in step_setting[cur_step][str(d)].keys():
            ret[d]['change'] = step_setting[cur_step][str(d)]['change']

    return ret


def cluster_avg():
    current_data = list(global_coll['concat_feat'].find({}))
    pred_df = pd.DataFrame(current_data)
    pred_df = pred_df.set_index('date_group')
    '''output'''
    ret = []
    feat_columns = ['school', 'grade', 'onlinetime', 'bindcash', 'totaltime',
               'combatscore', 'sex', 'deltatime']
    data_days = sorted(list(pred_df.index.value_counts().index), key=lambda x:int(x[3:]))
    res = pd.DataFrame(np.zeros([len(data_days), len(feat_columns)]), columns=feat_columns, index=data_days)
    for day in data_days:
        day_data = pred_df.loc[[day]]
        day_avg = day_data.mean(axis=0)
        res.loc[day] = day_avg
    res = res.reset_index()
    res = res.rename(columns={'index': 'date_group'})

    return res

def sp_train(is_norm=1):
    '''load csv'''
    trainset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'trainset.csv'), index_col=['uid', 'date_group'])
    testset = pd.read_csv(os.path.join(utils.get_churn_d1(), 'testset.csv'), index_col=['uid', 'date_group'])
    description = pd.read_csv(os.path.join(utils.get_churn_d1(), 'description_full.csv'), index_col='name')
    if is_norm:
        for col in trainset.columns:
            if col != 'churn':
                trainset[col] = (trainset[col].values - description.loc[col]['min']) / (
                            description.loc[col]['max'] - description.loc[col]['min'])
                testset[col] = (testset[col].values - description.loc[col]['min']) / (
                            description.loc[col]['max'] - description.loc[col]['min'])
    '''train data'''
    label = trainset['churn'].astype(np.int64)
    data = trainset.drop('churn', axis=1)
    sp_data = data[['deltatime', 'deltatime']]
    dataset = HousePrice(sp_data, label, cate_feat=[0], dtype=torch.long)
    dataloader = DataLoader(dataset,
                            batch_size=512,
                            collate_fn=dataset.collate_fn,
                            shuffle=True)
    '''test data'''
    test_label = testset['churn'].astype(np.int64)
    test_data = testset.drop('churn', axis=1)
    sp_test = test_data[['deltatime', 'deltatime']]
    test_dataset = HousePrice(sp_test, test_label, cate_feat=[0],
                              dtype=torch.long)  # ['school', 'grade', 'sex']
    test_loader = DataLoader(test_dataset,
                             batch_size=1024,
                             collate_fn=test_dataset.collate_fn,
                             shuffle=False)

    print('init dataset')

    in_dim = sp_data.shape[1]
    epoch = 100
    model_type = 'local'  # 'deepctr'

    if model_type == 'local':
        model = AutoInt(in_dim, out_dim=4, feat_size=[len(dataset.cate_feat), len(dataset.rest_feat)],
                        voc_size=60, emb_dim=256, head_num=4, n_layers=4).to('cuda')

    # elif model_type == 'deepctr':
    #     model = AutoInt_dc(linear_feature_columns=list(range(0, 3)),
    #                        dnn_feature_columns=list(range(3, in_dim)),
    #                        device='cuda').to('cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # , weight_decay=1e-3)
    loss_name = 'focal'
    if loss_name == 'topk':
        loss = topk_loss
    elif loss_name == 'focal':
        loss = focal_loss
    else:
        loss = nn.CrossEntropyLoss()

    best_loss = 10
    for epo in range(epoch):
        prob_all = []
        label_all = []
        loss_list = []
        for _data, _label in dataloader:
            if model_type == 'local':
                emb, output = model(_data)
            # elif model_type == 'deepctr':
            #     cat_data = torch.cat(_data, dim=1)
            #     emb, output = model(cat_data)
            _loss = loss(output, _label)
            loss_list.append(_loss.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            _loss.backward()
            optimizer.step()

            prob_all.append(output.detach().cpu().numpy())
            label_all.append(_label.detach().cpu().numpy())
        prob_all = np.concatenate(prob_all, axis=0)
        label_all = np.concatenate(label_all, axis=0)
        f1, auc, acc = print_metrics(prob_all, label_all)
        print('avg loss: %f, f1: %.4f, auc: %.4f, acc: %.4f' % (np.mean(loss_list).item(), f1, auc, acc))
        if test_loader and epo % 10 == 0:
            model.eval()
            tmp_list = []
            prob_all = []
            label_all = []
            for _data, _label in test_loader:
                if model_type == 'local':
                    _, output = model(_data)
                # elif model_type == 'deepctr':
                #     cat_data = torch.cat(_data, dim=1)
                #     _, output = model(cat_data)
                _loss = loss(output, _label)
                tmp_list.append(_loss.clone().detach().cpu().numpy())
                prob_all.append(output.detach().cpu().numpy())
                label_all.append(_label.detach().cpu().numpy())
            mean_loss = np.mean(tmp_list).item()
            prob_all = np.concatenate(prob_all, axis=0)
            label_all = np.concatenate(label_all, axis=0)
            f1, auc, acc = print_metrics(prob_all, label_all)
            print('VALIDATION\navg loss: %f, f1: %.4f, auc: %.4f, acc: %.4f' % (mean_loss, f1, auc, acc))
            # if mean_loss < best_loss:
            print('epo: %d, loss: %f' % (epo, mean_loss))
            # best_loss = mean_loss
            save_name = 'pretrain_epo_%d_%.4f.ckpt'
            if is_norm:
                save_name = 'norm_' + save_name
            # save_model(model, optimizer, save_name % (epo, mean_loss))
            model.train()

def highest_shap():
    data = pd.DataFrame(list(global_coll['shap_churn'].find({}))).drop('_id', axis=1)
    df_mean = data.mean(axis=0)
    print(1)

if __name__ == '__main__':
    # features = pd.read_csv(os.path.join(utils.get_churn_d1(), 'concat_feat.csv'), index_col=['uid', 'date_group'])
    # features = features.drop(['class'], axis=1)
    # features = features.reset_index()
    # features.to_csv(os.path.join(utils.get_churn_d1(), 'fullset.csv'), index=False)
    '''autoint'''
    # graph_metrics_process()
    # update_portrait()
    # concat_all_feature()
    train(is_norm=1,ablation='woportrait')
    # inference(1, 'norm_focal_pretrain_epo_20_0.1554_wosocial.ckpt', 'wosocial')
    # inference(0, '_pretrain_epo_10_1.2975.ckpt')
    '''to final data'''
    # save_pred()
    # add_class()
    # save_shap()
    # update_shap()
    '''counterfactual'''
    # makeDescription('testset.csv','description_test.csv')
    # makeDescription('fullset.csv','description_full.csv')
    # counter_factual_example({"cf_range":{"school":{"min":3,"max":5}}})
    # counter_factual_example({})
    '''clustering'''
    # cluster_avg()
    '''check'''
    # check()
    # highest_shap()
