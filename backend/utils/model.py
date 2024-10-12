import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from cf_ml.utils.dir_manager import DirectoryManager
from utils import utils
import os

class Interact_Layer(nn.Module):
    def __init__(self, in_dim, feat_dim=8, w_dim=32, out_dim=32):
        super(Interact_Layer, self).__init__()
        self.in_dim = in_dim
        self.feat_dim = feat_dim
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.wq = nn.Linear(in_dim, w_dim, bias=False)
        self.wk = nn.Linear(in_dim, w_dim, bias=False)
        self.wv = nn.Linear(in_dim, in_dim, bias=False)
        self.res = nn.Linear(in_dim*feat_dim, out_dim, bias=False)

    def forward(self, emb):
        eq, ek, ev = self.wq(emb), self.wk(emb), self.wv(emb)
        qk = eq @ ek.transpose(-1, -2) / self.feat_dim
        attn = torch.softmax(qk, dim=-1)
        attn_emb = attn @ ev
        emb = (attn_emb + emb).view(attn_emb.shape[0], -1)
        res_emb = torch.relu(self.res(emb))
        return res_emb


class AutoInt(nn.Module):
    def __init__(self, in_dim, emb_dim=8, out_dim=2, feat_size=[0, 0],
                 n_layers=3, voc_size=11, head_num=2, dataset=None, split=None, flag=0):
        super(AutoInt, self).__init__()
        '''cf'''
        if dataset is not None:
            self._dataset = dataset
            self._features = self._dataset.dummy_features
            self._target = self._dataset.dummy_target
            self._prediction = "{}_pred".format(self._dataset.target)
            root_dit = os.path.join(utils.get_churn_d1(), 'cf_output')
            self.dir_manager = DirectoryManager(dataset, 'AutoInt', root=root_dit)
        self._split = split
        self.flag = flag

        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.feat_size = feat_size # sparse first, dense second
        self.n_layers = n_layers
        self.voc_size = voc_size
        self.init_dim = 8

        self.sparse_size = feat_size[0]
        self.dense_size = feat_size[1]
        self.sparse_emb = nn.ModuleList([nn.Embedding(self.voc_size, self.init_dim) for i in range(self.sparse_size)])
        self.dense_emb = nn.ModuleList([nn.Linear(1, self.init_dim) for i in range(self.dense_size)])
        self.conv2 = nn.Conv1d(in_channels=self.init_dim, out_channels=self.init_dim, kernel_size=5, stride=1, padding=2)
        '''Interact_Layer'''
        # self.interact = Interact_Layer(self.init_dim, feat_dim=self.sparse_size+self.dense_size, w_dim=self.in_dim, out_dim=emb_dim)
        # self.fc1 = nn.Linear(emb_dim, self.out_dim)
        '''InteractingLayer'''
        self.interact = nn.ModuleList([InteractingLayer(self.init_dim, head_num=head_num, device='cuda') for _ in range(n_layers)])
        self.fc1 = nn.Linear(in_dim*self.init_dim, self.out_dim)

    def forward(self, x):
        if type(x) == torch.Tensor:
            spar = x[:, :self._split]
            dens = x[:, self._split:]
        elif type(x) == list:
            spar, dens = x
        elif type(x) == np.ndarray:
            x = torch.FloatTensor(x)
            spar = x[:, :self._split]
            dens = x[:, self._split:]
        else:
            raise NotImplementedError
        spar, dens = spar.to('cuda'), dens.to('cuda')
        spar_min = torch.min(spar).detach()
        if spar_min < 0:
            spar = spar.detach() - spar_min
        assert torch.max(spar) < self.voc_size, torch.max(spar)
        spar_feat = torch.stack([emb_fun(spar[:, i].long()) for i, emb_fun in enumerate(self.sparse_emb)], dim=1)
        dens_feat = torch.stack([emb_fun(dens[:, i].unsqueeze(1)) for i, emb_fun in enumerate(self.dense_emb)], dim=1)
        cat_feat = torch.cat([spar_feat, dens_feat], dim=1)
        infeat = cat_feat#self.conv2(cat_feat.transpose(-1,-2)).transpose(-1,-2)
        # infeat = infeat / (infeat.norm(dim=2).mean(dim=0)[None, :, None]+1)
        '''interacting in deepctr'''
        for _layer in self.interact:
            infeat = _layer(infeat)
        emb = torch.flatten(infeat, start_dim=1)
        out = torch.softmax(self.fc1(emb),dim=1)
        if self.flag == 1:
            return out
        elif self.flag == 2:
            return out.cpu().detach().numpy()
        else:
            return emb, out

    def report(self, x, y=None, preprocess=True):
        """Generate the report from the feature values and target values (optional).
        The report includes (features, target, prediction)."""
        if preprocess:
            x = self._dataset.preprocess_X(x)
            # if y is not None:
            #     y = self._dataset.preprocess_y(y)

        if isinstance(x, pd.DataFrame):
            x = x[self._features].values
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.cuda()
        pred = self.forward(x).cpu().detach().numpy()
        if y is None:
            y = np.zeros(pred.shape)
        if y.shape[1] != 1:
            y = np.argmax(y, axis=1)[:, None]
            pred = np.argmax(pred, axis=1)[:, None]

        x = x.cpu().numpy()
        report_df = self._dataset.inverse_preprocess(np.concatenate((x, y), axis=1))
        report_df[self._prediction] = self._dataset.inverse_preprocess_y(pred)

        return report_df

'''DeepCTR'''
class InteractingLayer(nn.Module):
    """A Layer used in AutoInt that model the correlations between different feature fields by multi-head self-attention mechanism.
      Input shape
            - A 3D tensor with shape: ``(batch_size,field_size,embedding_size)``.
      Output shape
            - 3D tensor with shape:``(batch_size,field_size,embedding_size)``.
      Arguments
            - **in_features** : Positive integer, dimensionality of input features.
            - **head_num**: int.The head number in multi-head self-attention network.
            - **use_res**: bool.Whether or not use standard residual connections before output.
            - **seed**: A Python integer to use as random seed.
      References
            - [Song W, Shi C, Xiao Z, et al. AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks[J]. arXiv preprint arXiv:1810.11921, 2018.](https://arxiv.org/abs/1810.11921)
    """

    def __init__(self, embedding_size, head_num=2, use_res=True, scaling=False, seed=1024, device='cpu'):
        super(InteractingLayer, self).__init__()
        if head_num <= 0:
            raise ValueError('head_num must be a int > 0')
        if embedding_size % head_num != 0:
            raise ValueError('embedding_size is not an integer multiple of head_num!')
        self.att_embedding_size = embedding_size // head_num
        self.head_num = head_num
        self.use_res = use_res
        self.scaling = scaling
        self.seed = seed

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

        self.to(device)

    def forward(self, inputs):

        if len(inputs.shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(inputs.shape)))

        # None F D
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        # head_num None F D/head_num
        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)  # head_num None F F
        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)  # head_num None F F
        result = torch.matmul(self.normalized_att_scores, values)  # head_num None F D/head_num

        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)  # None F D
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)

        return result

# class AutoInt_dc(nn.Module):
#     """Instantiates the AutoInt Network architecture.
#
#     :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
#     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
#     :param att_layer_num: int.The InteractingLayer number to be used.
#     :param att_head_num: int.The head number in multi-head  self-attention network.
#     :param att_res: bool.Whether or not use standard residual connections before output.
#     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
#     :param dnn_activation: Activation function to use in DNN
#     :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
#     :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
#     :param dnn_use_bn:  bool. Whether use BatchNormalization before activation or not in DNN
#     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
#     :param init_std: float,to use as the initialize std of embedding vector
#     :param seed: integer ,to use as random seed.
#     :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
#     :param device: str, ``"cpu"`` or ``"cuda:0"``
#     :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
#     :return: A PyTorch model instance.
#
#     """
#
#     def __init__(self, linear_feature_columns, dnn_feature_columns, att_layer_num=3,
#                  att_head_num=2, att_res=True, dnn_hidden_units=(256, 128), dnn_activation='relu',
#                  l2_reg_dnn=0, l2_reg_embedding=1e-5, dnn_use_bn=False, dnn_dropout=0, init_std=0.0001, seed=1024,
#                  task='binary', device='cpu', gpus=None):
#
#         super(AutoInt_dc, self).__init__()
#         if len(dnn_hidden_units) <= 0 and att_layer_num <= 0:
#             raise ValueError("Either hidden_layer or att_layer_num must > 0")
#         self.use_dnn = len(dnn_feature_columns) > 0 and len(dnn_hidden_units) > 0
#         field_num = len(self.embedding_dict)
#
#         embedding_size = self.embedding_size
#
#         if len(dnn_hidden_units) and att_layer_num > 0:
#             dnn_linear_in_feature = dnn_hidden_units[-1] + field_num * embedding_size
#         elif len(dnn_hidden_units) > 0:
#             dnn_linear_in_feature = dnn_hidden_units[-1]
#         elif att_layer_num > 0:
#             dnn_linear_in_feature = field_num * embedding_size
#         else:
#             raise NotImplementedError
#
#         self.dnn_linear = nn.Linear(dnn_linear_in_feature, 1, bias=False).to(device)
#         self.dnn_hidden_units = dnn_hidden_units
#         self.att_layer_num = att_layer_num
#         if self.use_dnn:
#             self.dnn = DNN(self.compute_input_dim(dnn_feature_columns), dnn_hidden_units,
#                            activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
#                            init_std=init_std, device=device)
#             self.add_regularization_weight(
#                 filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], self.dnn.named_parameters()), l2=l2_reg_dnn)
#         self.int_layers = nn.ModuleList(
#             [InteractingLayer(embedding_size, att_head_num, att_res, device=device) for _ in range(att_layer_num)])
#
#         self.to(device)
#
#     def forward(self, X):
#
#         sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
#                                                                                   self.embedding_dict)
#         logit = self.linear_model(X)
#
#         att_input = concat_fun(sparse_embedding_list, axis=1)
#
#         for layer in self.int_layers:
#             att_input = layer(att_input)
#
#         att_output = torch.flatten(att_input, start_dim=1)
#
#         dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
#
#         if len(self.dnn_hidden_units) > 0 and self.att_layer_num > 0:  # Deep & Interacting Layer
#             deep_out = self.dnn(dnn_input)
#             stack_out = concat_fun([att_output, deep_out])
#             logit += self.dnn_linear(stack_out)
#         elif len(self.dnn_hidden_units) > 0:  # Only Deep
#             deep_out = self.dnn(dnn_input)
#             logit += self.dnn_linear(deep_out)
#         elif self.att_layer_num > 0:  # Only Interacting Layer
#             logit += self.dnn_linear(att_output)
#         else:  # Error
#             pass
#
#         y_pred = self.out(logit)
#
#         return y_pred

'''dataset'''
class HousePrice(Dataset):
    def __init__(self, data, label, dtype=torch.float64, cate_feat=[]):
        super(HousePrice, self).__init__()
        # self.index = data.index.values.tolist()
        self.data = data
        self.label = label
        self._dtype = dtype
        self.cate_feat = cate_feat
        self.rest_feat = list(set(list(range(len(data.columns))))-set(self.cate_feat))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        if type(index) != slice:
            index = [index]
        return torch.cat([torch.FloatTensor(self.data.iloc[index].values),
                          torch.FloatTensor(self.label.iloc[index].values).unsqueeze(1)], dim=1)

    def get_mulitem(self, index):
        if index == []:
            return []
        data = self.data.loc[index]
        label = self.label.loc[index]
        return HousePrice(data, label, cate_feat=self.cate_feat)

    def collate_fn(self, data):
        if type(data) == list:
            batch = torch.cat(data, dim=0)
        else:
            batch = data[0]
        batch_data = batch[:, :-1].to('cuda')
        batch_label = torch.Tensor(batch[:, -1]).type(self._dtype).to('cuda')
        if len(self.cate_feat) != 0:
            batch_data = [batch_data[:, self.cate_feat].to('cuda'), batch_data[:, self.rest_feat].to('cuda')]
        return batch_data, batch_label

    def get_dim(self):
        return self.data.shape[1]