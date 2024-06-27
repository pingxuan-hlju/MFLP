import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module

from layers.att_layers import DenseAtt


def get_dim_act_curv(args):

    if not args.act:
        act = lambda x: x
    else:
        act = getattr(F, args.act)
    acts = [act] * (args.num_layers - 1)
    dims = [args.feat_dim] + ([args.dim] * (args.num_layers - 1))
    dims += [args.dim]
    acts += [act]
    n_curvatures = args.num_layers
    if args.c is None:
        # create list of trainable curvature parameters
        curvatures = [nn.Parameter(torch.Tensor([1.])) for _ in range(n_curvatures)]
    else:
        # fixed curvature
        curvatures = [torch.tensor([args.c]) for _ in range(n_curvatures)]
        if not args.cuda == -1:
            curvatures = [curv.to(args.cuda) for curv in curvatures]
    return dims, acts, curvatures


class HyperbolicGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self,manifold, in_features, out_features, c_in, c_out, dropout, act, use_bias, use_att, local_agg):
        super(HyperbolicGraphConvolution, self).__init__()
        self.linear = HypLinear(manifold, in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg(manifold, c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = HypAct(manifold, c_in, c_out, act)


    def forward(self, input):
        x, adj = input
        if x.shape[1] == 1546:
            liner_h = self.linear.forward(x)
        else:
            liner_h = x
        agg_h = self.agg.forward(liner_h, adj)
        h = self.hyp_act.forward(agg_h,liner_h)
        output = h, adj
        return output

class HypLinear(nn.Module):

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features  # 1546
        self.out_features = out_features  # 64
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias_drug = nn.Parameter(torch.Tensor(out_features))
        self.bias_micr = nn.Parameter(torch.Tensor(out_features))
        # 初始化可学习的权重向量
        self.weight_vector_drug = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_vector_microbe = nn.Parameter(torch.Tensor(out_features, in_features))
        # self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight_vector_drug, gain=math.sqrt(2))
        init.xavier_uniform_(self.weight_vector_microbe, gain=math.sqrt(2))
        init.constant_(self.bias_drug, 0)
        init.constant_(self.bias_micr, 0)



    def forward(self, x):
        drop_drug_weight = F.dropout(self.weight_vector_drug, self.dropout, training=self.training).cuda()
        drop_micr_weight = F.dropout(self.weight_vector_microbe, self.dropout, training=self.training).cuda()
        mv_drug = self.manifold.mobius_matvec(drop_drug_weight, x, self.c)
        mv_micr = self.manifold.mobius_matvec(drop_micr_weight, x, self.c)
        res_drug = self.manifold.proj(mv_drug, self.c)
        res_micr = self.manifold.proj(mv_micr, self.c)
        # mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        if self.use_bias:
            bias_drug = self.manifold.proj_tan0(self.bias_drug.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias_drug, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res_drug = self.manifold.mobius_add(res_drug, hyp_bias, c=self.c)
            res_drug = self.manifold.proj(res_drug, self.c)
            bias_micr = self.manifold.proj_tan0(self.bias_micr.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias_micr, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res_micr = self.manifold.mobius_add(res_micr, hyp_bias, c=self.c)
            res_micr = self.manifold.proj(res_micr, self.c)
        drug_feature = res_drug
        drug_feature = drug_feature[:1373, :]
        micr_feature = res_micr
        micr_feature = micr_feature[-173:, :]
        res = torch.cat((drug_feature, micr_feature),dim=0)
#         print(res.shape)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )


class HypAgg(Module):

    def __init__(self, manifold, c, in_features, dropout, use_att, local_agg):
        super(HypAgg, self).__init__()
        self.manifold = manifold
        self.c = c

        self.in_features = in_features
        self.dropout = dropout
        self.local_agg = local_agg
        self.use_att = use_att
        if self.use_att:
            self.att = DenseAtt(in_features, dropout)

    def forward(self, x, adj):
        x_tangent = self.manifold.logmap0(x, c=self.c).to(torch.float32)
        if self.use_att:
            if self.local_agg:
                # if hasattr(torch.cuda, 'empty_cache'):
                #     torch.cuda.empty_cache()
                x_local_tangent = []
                for i in range(x.size(0)):
                    x_local_tangent.append(self.manifold.logmap(x[i], x, c=self.c))
                x_local_tangent = torch.stack(x_local_tangent, dim=0)
                adj_att = self.att(x_tangent, adj)
                att_rep = adj_att.unsqueeze(-1) * x_local_tangent
                # 公式9中的累加操作
                support_t = torch.sum(adj_att.unsqueeze(-1) * x_local_tangent, dim=1)
                output = self.manifold.proj(self.manifold.expmap(x, support_t, c=self.c), c=self.c)
                return output
            else:
                adj_att = self.att(x_tangent, adj)
                ### 2023.3.26改.to(torch.double)
                support_t = torch.matmul(adj_att, x_tangent.to(torch.double))
        else:
            adj = adj.to(torch.float32)
            support_t = torch.spmm(adj, x_tangent)
        output = self.manifold.proj(self.manifold.expmap0(support_t, c=self.c), c=self.c)
        return output

    def extra_repr(self):
        return 'c={}'.format(self.c)


class HypAct(Module):

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act
        self.weightnode = nn.Parameter(torch.Tensor(128, 64))
        self.biasnode = nn.Parameter(torch.Tensor(1546,1))
        init.xavier_uniform_(self.weightnode, gain=math.sqrt(2))
        init.constant_(self.biasnode, 0)

    def forward(self, x, liner_h_last):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        result = self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)
        ###########20240325加，初试无att###################################################
        zf = torch.cat((x, liner_h_last), dim=1)
        delt1 = zf.to(torch.double) @ self.weightnode.to(torch.double) + self.biasnode
        delt1 = torch.relu(delt1)
        result = torch.mul(delt1, x) + torch.mul((1 - delt1), liner_h_last)
        ##############################################################
        return result

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
