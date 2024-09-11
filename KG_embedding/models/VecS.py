
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection, euc_sqdistance
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, hyp_distance

GIE_MODELS = ["VecS"]


class BaseH(KGModel):

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        # 初始化实体和关系的embedding
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init1 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
            c_init2 = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
            c_init1 = torch.ones((1, 1), dtype=self.data_type)
            c_init2 = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)
        self.c1= nn.Parameter(c_init1, requires_grad=True)
        self.c2 = nn.Parameter(c_init2, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])
        
    def get_euluc_rhs(self, eulucclass_idx):
        return self.entity(eulucclass_idx), self.bt(eulucclass_idx)


    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2
    
    def compute_embedding(self, examples, idx2eulucclass, batch_size=500):
        """Compute ranking-based evaluation metrics.
        
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores
        """
        queries = examples.clone()
        results = np.zeros(len(queries))
        all_embeddings = torch.zeros(len(queries), self.rank)
        with torch.no_grad():
            b_begin = 0
            # candidates = self.get_rhs(queries, eval_mode=True) # all entities embedding
            eulucclass_idx = torch.LongTensor(list(idx2eulucclass.keys())).cuda()
            euluc_candidates = self.get_euluc_rhs(eulucclass_idx)
            
            # make idx2zeroindex
            idx_list = list(idx2eulucclass.keys())
            zeroindex2idx = {}
            for i in range(len(idx_list)):
                zeroindex2idx[i] = idx_list[i]
                
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)

                # scores = self.score(q, candidates, eval_mode=True) # all entities score
                euluc_scores = self.score(q, euluc_candidates, eval_mode=True) # all entities score

                results[b_begin:b_begin + batch_size] = np.array([zeroindex2idx[idx] for idx in torch.argmax(euluc_scores, dim=1).cpu().numpy()])
                all_embeddings[b_begin:b_begin + batch_size] = q[0][0].cpu()
                b_begin += batch_size
                
            label = queries[:, 2]
            wrong_index = np.where(results != label.cpu().numpy())[0]
            
            wrong_examples = queries[wrong_index]
            wrong_label = results[wrong_index]

        return results, all_embeddings, wrong_examples, wrong_label

class VecS(BaseH):

    def __init__(self, args, relation_type_index):
        super(VecS, self).__init__(args)

        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)  # 旋转矩阵（可学习）(为什么是2*rank？)
        self.rel_diag1 = nn.Embedding(self.sizes[1], self.rank)  # 旋转矩阵（可学习）
        self.rel_diag2 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag3 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag4 = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag3.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.rel_diag4.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        self.relation_type_index = relation_type_index
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_index(self, queries):
        vec_queries_index = None
        seman_queries_index = None
        vec_seman_queries_index = None
        for index in self.relation_type_index['vec']:
            queries_index = (queries[:, 1] == index) | (queries[:, 1] == index + self.sizes[1] // 2)
            if vec_queries_index is None:
                vec_queries_index = queries_index
            else:
                vec_queries_index = vec_queries_index | queries_index
        for index in self.relation_type_index['seman']:
            queries_index = (queries[:, 1] == index) | (queries[:, 1] == index + self.sizes[1] // 2)
            if seman_queries_index is None:
                seman_queries_index = queries_index
            else:
                seman_queries_index = seman_queries_index | queries_index
        for index in self.relation_type_index['vec_seman']:
            queries_index = (queries[:, 1] == index) | (queries[:, 1] == index + self.sizes[1] // 2)
            if vec_seman_queries_index is None:
                vec_seman_queries_index = queries_index
            else:
                vec_seman_queries_index = vec_seman_queries_index | queries_index
        return vec_queries_index, seman_queries_index, vec_seman_queries_index

    def get_queries(self, queries):

        vec_head = self.entity(queries[:, 0])
        vec_res = givens_rotations(self.rel_diag3(queries[:, 1]), vec_head) + self.rel(queries[:, 1])

        # 同理
        vec_seman_c = F.softplus(self.c2[queries[:, 1]])
        vec_seman_head = expmap0(self.entity(queries[:, 0]), vec_seman_c)
        vec_seman_rel = self.rel(queries[:, 1])
        vec_seman_rel = expmap0(vec_seman_rel, vec_seman_c)
        vec_seman_lhs = project(mobius_add(vec_seman_head, vec_seman_rel, vec_seman_c), vec_seman_c)
        vec_seman_res = givens_rotations(self.rel_diag4(queries[:, 1]), vec_seman_lhs)

        seman_head = self.entity(queries[:, 0])  # 原始实体嵌入
        # rot_mat, _ = torch.chunk(self.rel_diag(seman_queries[:, 1]), 2, dim=1)
        # seman_res = givens_rotations(rot_mat, seman_head)
        seman_rel = self.rel(queries[:, 1])
        seman_res = seman_head + seman_rel


        c = F.softplus(self.c[queries[:, 1]])
        cands = torch.cat([vec_res.view(-1, 1, self.rank), vec_seman_res.view(-1, 1, self.rank), seman_res.view(-1, 1, self.rank)], dim=1)
        # cands = torch.cat([vec_res.view(-1, 1, self.rank), seman_res.view(-1, 1, self.rank)], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel = self.rel(queries[:, 1])
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)

        return (res, c), self.bh(queries[:, 0])