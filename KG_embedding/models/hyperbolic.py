"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import os
from models.base import KGModel
from utils.euclidean import givens_rotations, givens_reflection
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c


HYP_MODELS = ["RotH", "RefH", "AttH"]

class BaseH(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseH, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], self.rank), dtype=self.data_type) - 1.0
        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])
        
    def get_euluc_rhs(self, eulucclass_idx):
        return self.entity(eulucclass_idx), self.bt(eulucclass_idx)

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
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

        return results, all_embeddings


class RotH(BaseH):
    """Hyperbolic 2x2 Givens rotations"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])


class RefH(BaseH):
    """Hyperbolic 2x2 Givens reflections"""

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        lhs = givens_reflection(self.rel_diag(queries[:, 1]), self.entity(queries[:, 0]))
        lhs = expmap0(lhs, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])


class AttH(BaseH):
    """Hyperbolic attention model combining translations, reflections and rotations"""

    def __init__(self, args):
        super(AttH, self).__init__(args)
        self.rel_diag = nn.Embedding(self.sizes[1], 2 * self.rank)
        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 2 * self.rank), dtype=self.data_type) - 1.0
        self.context_vec = nn.Embedding(self.sizes[1], self.rank)
        self.context_vec.weight.data = self.init_size * torch.randn((self.sizes[1], self.rank), dtype=self.data_type)
        self.act = nn.Softmax(dim=1)
        if args.dtype == "double":
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).double().cuda()
        else:
            self.scale = torch.Tensor([1. / np.sqrt(self.rank)]).cuda()

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = self.entity(queries[:, 0])
        rot_mat, ref_mat = torch.chunk(self.rel_diag(queries[:, 1]), 2, dim=1)
        rot_q = givens_rotations(rot_mat, head).view((-1, 1, self.rank))
        ref_q = givens_reflection(ref_mat, head).view((-1, 1, self.rank))
        cands = torch.cat([ref_q, rot_q], dim=1)
        context_vec = self.context_vec(queries[:, 1]).view((-1, 1, self.rank))
        att_weights = torch.sum(context_vec * cands * self.scale, dim=-1, keepdim=True)
        att_weights = self.act(att_weights)
        att_q = torch.sum(att_weights * cands, dim=1)
        lhs = expmap0(att_q, c)
        rel, _ = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel = expmap0(rel, c)
        res = project(mobius_add(lhs, rel, c), c)
        return (res, c), self.bh(queries[:, 0])

class Mutiview_RotH(BaseH):

    def __init__(self, args):
        super(Mutiview_RotH, self).__init__(args)
        graph_level_c_init = torch.ones((1, 1), dtype=self.data_type)
        self.graph_level_c = nn.Parameter(graph_level_c_init, requires_grad=True)

        self.rel_level_c = self.c

        ndoe_level_c_init = torch.ones((self.sizes[0], 1), dtype=self.data_type)
        self.node_level_c = nn.Parameter(ndoe_level_c_init, requires_grad=True)

        ## 统一表征空间

    def get_rel_level_queries(self, queries, curvatue):
        """Compute embedding and biases of queries."""
        c = F.softplus(curvatue[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])

    def get_node_level_queries(self, queries, curvatue):
        """Compute embedding and biases of queries."""
        c = F.softplus(curvatue[queries[:, 0]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])

    def get_graph_level_queries(self, queries, curvatue):
        """Compute embedding and biases of queries."""
        c = F.softplus(curvatue[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = project(mobius_add(head, rel1, c), c)
        res1 = givens_rotations(self.rel_diag(queries[:, 1]), lhs)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])

    def get_queries(self, queries):

        node_level_lhs_e, node_level_lhs_biases = self.get_node_level_queries(queries, self.node_level_c)
        rel_level_lhs_e, rel_level_lhs_biases = self.get_rel_level_queries(queries, self.rel_level_c)
        graph_level_lhs_e, graph_level_lhs_biases = self.get_graph_level_queries(queries, self.graph_level_c)

        return
