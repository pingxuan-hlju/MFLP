import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.layers import FermiDiracDecoder
import layers.hyp_layers as hyp_layers
import manifolds

class GHGNN():
    """
    Hyperbolic-GCN.
    """

    def __init__(self, args):
        super(GHGNN, self).__init__()
        # self.layer1_dim = args.layer1_dim # 128
        # self.layer2_dim = args.layer2_dim # 64
        # self.drug_nums = args.drug_nums # 1373
        # self.mirc_nums = args.mirc_nums # 173
        self.manifold_name = args.manifold # Hyperboloid
        self.decoder = FermiDiracDecoder(args.r, args.t)
        self.c = nn.Parameter(torch.Tensor([1.]))
        if self.manifold_name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        # self.nodes = args.drug_nums + args.mirc_nums # 1546
        assert args.num_layers > 1
        dims, acts, self.curvatures = hyp_layers.get_dim_act_curv(args)
        self.curvatures.append(self.c)
        hgc_layers = []
        for i in range(len(dims) - 1):
            c_in, c_out = self.curvatures[i], self.curvatures[i + 1]
            in_dim, out_dim = dims[i], dims[i + 1]
            act = acts[i]
            hgc_layers.append(
                hyp_layers.HyperbolicGraphConvolution(
                    self.manifold_name, in_dim, out_dim, c_in, c_out, args.dropout, act, args.bias, args.use_att,
                    args.local_agg
                )
            )
        self.layers = nn.Sequential(*hgc_layers)
        self.encode_graph = True

    def encoder(self, x, adj):
        save = []
        if self.encode_graph:
            input = (x, adj, save)
            output, _, save = self.layers.forward(input)
        else:
            output = self.layers.forward(x)
        return output, save

    def decoder(self, save, idx):
        h1 = save[0]
        h2 = save[1]
        # emb_in = h1[idx[:, 0], :]
        # emb_out = h1[idx[:, 1] + 1373, :]
        # sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        # probs1 = self.dc.forward(sqdist)
        # (1546, 256)
        h = torch.cat((h1, h2), dim=1)
        emb_left, emb_right = h[idx[:, 0]], h[idx[:, 1] + 1373]
        emb = torch.cat((emb_left, emb_right), dim=1)
        return self.decoder(emb).squeeze(-1)

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.c = args.c
        self.manifold_name = args.manifold
        self.encoder = GHGNN.encoder(args)
        self.decoder = FermiDiracDecoder(args.r, args.t)
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1


    def forward(self, x, adj ,idx):
        # 这里是特征映射
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(self.x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        x_tan = self.manifold_name.proj_tan0(x, self.curvatures[0])
        x_hyp = self.manifold_name.expmap0(x_tan, c=self.curvatures[0])
        x_hyp = self.manifold_name.proj(x_hyp, c=self.curvatures[0])
        # 映射好的特征进入GHGNN
        output, save = self.encoder(x_hyp, adj)
        preds = self.decoder(save, idx)
        return preds

from config import parser
args = parser.parse_args()
model = Model(args)
print(model)