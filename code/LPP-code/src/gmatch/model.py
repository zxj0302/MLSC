import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch_scatter import scatter
from math import sqrt
from gmatch.training import ModelWraper

mse_loss = MSELoss()

def w_index_to_scatter_index(w_index):
    scatter_w_index = torch.zeros_like(w_index)
    scatter_count = 0
    for i, v in enumerate(w_index):
        if i == 0: continue
        if w_index[i] == w_index[i-1]:
            scatter_w_index[i] = scatter_count
        else:
            scatter_count = scatter_count + 1
            scatter_w_index[i] = scatter_count
    return scatter_w_index

def whole_graph_error(preds, labels, w_index):
    scatter_index = w_index_to_scatter_index(w_index)
    preds_reduce = scatter(preds, scatter_index, reduce='sum', dim=0)
    labels_reduce = scatter(labels, scatter_index, reduce='sum', dim=0)
    w_label_mean = labels_reduce.mean()
    w_mse = (preds_reduce - labels_reduce).square().mean()
    w_rmse = w_mse.sqrt()
    return w_mse, w_rmse, w_label_mean

class LPPModelWraper(ModelWraper):
    def compute(self, batch):
        vecs, lengths, labels, w_index = batch
        # process scatter_indexs
        scatter_indexs = []
        for i, l in enumerate(lengths):
            scatter_indexs.append( torch.ones(l, dtype=torch.int64, device=labels.device) * i )
        scatter_indexs = torch.cat(scatter_indexs)
        
        preds = self.model((vecs, scatter_indexs)) # forward computation
        preds = preds.squeeze(1)

        loss = mse_loss(preds, labels)
        
        w_mse, w_rmse, w_label_mean = whole_graph_error(preds, labels, w_index)

        step_results = {
            'loss': loss,
            'rmse': loss.sqrt(),
            'label_mean': labels.mean(),
            'w_mse': w_mse,
            'w_rmse': w_rmse,
            'w_label_mean': w_label_mean
        }

        return step_results
    pass


class LPPTransformer(nn.Module):
    def __init__(self, max_nodes):
        super(LPPTransformer, self).__init__()
        from torch.nn import TransformerEncoderLayer
        self.max_nodes = max_nodes
        self.feature_dim = max_nodes**2
        self.linear_proj = nn.Linear(self.feature_dim, 128, bias=False)
        self.encoder_layer = TransformerEncoderLayer(d_model=128, nhead=4, dropout=0.0, batch_first=True)
        self.fc = nn.Linear(128, 1)
    
    def process_vectors(self, vectors, scatter_indexs, mode='pad'):
        lengths = []
        last_position = 0
        for i, v in enumerate(scatter_indexs):
            if i == 0: continue
            if v == scatter_indexs[i-1]:
                continue
            else:
                lengths.append(i - last_position)
                last_position = i
        lengths.append( len(scatter_indexs) - last_position )
        pad_num_words = max(lengths)

        if mode == 'pad':
            padded_vecs = []
            start = 0
            for l in lengths:
                end = start + l
                vec = vectors[start:end]
                if l < pad_num_words:
                    pad = torch.zeros((pad_num_words-l, vec.shape[1]), device=vec.device, dtype=vec.dtype)
                    vec = torch.cat([vec, pad], dim=0)
                start = end
                padded_vecs.append(vec)
            padded_vecs = torch.stack(padded_vecs, dim=0)
            return padded_vecs
        elif mode == 'recover':
            assert vectors.shape[0] == len(lengths)
            recover = []
            for i, vec in enumerate(vectors):
                recover.append( vec[:lengths[i]] )
            recover = torch.cat(recover, dim=0)
            return recover
        

    def forward(self, inputs): 
        vecs, scatter_indexs = inputs
        vecs = self.linear_proj(vecs)
        vecs = self.process_vectors(vecs, scatter_indexs, mode='pad')
        out = self.encoder_layer(vecs)
        out = self.process_vectors(out, scatter_indexs, mode='recover')
        pred = self.fc(out)
        pred = scatter(pred, scatter_indexs, reduce='mean', dim=0)
        return pred


class LPPCNN(nn.Module):
    def __init__(self, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.in_channels = 1
        self.out_channels = 4
        self.conv21 = nn.Conv2d(self.in_channels, self.out_channels, 2)
        self.conv22 = nn.Conv2d(self.out_channels, self.out_channels, 2)
        self.conv31 = nn.Conv2d(self.in_channels, self.out_channels, 3)
        self.conv32 = nn.Conv2d(self.out_channels, self.out_channels, 3)
        self.conv41 = nn.Conv2d(self.in_channels, self.out_channels, 4)
        self.conv42 = nn.Conv2d(self.out_channels, self.out_channels, 4)

        outdim_k2 = max(max_nodes, 7) - (2-1)*2
        outdim_k3 = max(max_nodes, 7) - (3-1)*2
        outdim_k4 = max(max_nodes, 7) - (4-1)*2
        self.fc1 = nn.Linear(self.out_channels * (outdim_k2**2 + outdim_k3**2 + outdim_k4**2) , 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 1)

    def process_vectors(self, vectors):
        new_vecs = []
        pad_size = max(self.max_nodes, 7) # 7 for the conv42(conv41) to be corrently operated
        for vec in vectors:
            vec = vec[vec!=-1]
            size = int(sqrt(len(vec)))
            vec = vec.reshape(( size, size))
            pad_s = pad_size - size
            vec = torch.nn.functional.pad(vec, pad=(0, pad_s, 0, pad_s), mode='constant', value=-1)
            new_vecs.append(vec)
        new_vecs = torch.stack(new_vecs)
        new_vecs = torch.unsqueeze(new_vecs, dim=1)
        return new_vecs

    def forward(self, inputs):
        vecs, scatter_indexs = inputs
        vecs = self.process_vectors(vecs)
        x_k2 = self.conv22(F.relu(self.conv21(vecs)))
        x_k3 = self.conv32(F.relu(self.conv31(vecs)))
        x_k4 = self.conv42(F.relu(self.conv41(vecs)))
        x_k2 = torch.flatten(x_k2, 1)
        x_k3 = torch.flatten(x_k3, 1)
        x_k4 = torch.flatten(x_k4, 1)
        x = torch.cat([x_k2, x_k3, x_k4], axis=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        preds = self.fc3(x)
        preds = scatter(preds, scatter_indexs, reduce="mean", dim=0)
        return preds


class LPPMLP(nn.Module):
    def __init__(self, in_features, out_features, mid_features):
        super(LPPMLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mid_features = mid_features
        self.build_model()

    def build_model(self):
        self.mlp = nn.Sequential(
            nn.Linear(self.in_features, self.mid_features),
            nn.ReLU(),
            nn.Linear(self.mid_features, self.out_features),
        )

    def forward(self, inputs):
        vecs, scatter_indexs = inputs
        preds = self.mlp(vecs)
        preds = scatter(preds, scatter_indexs, reduce="mean", dim=0)
        return preds