"""Base Knowledge Graph embedding model."""
from abc import ABC, abstractmethod

import torch
from torch import nn
import numpy as np

class KGModel(nn.Module, ABC):
    """Base Knowledge Graph Embedding model class.

    Attributes:
        sizes: Tuple[int, int, int] with (n_entities, n_relations, n_entities)
        rank: integer for embedding dimension
        dropout: float for dropout rate
        gamma: torch.nn.Parameter for margin in ranking-based loss
        data_type: torch.dtype for machine precision (single or double)
        bias: string for whether to learn or fix bias (none for no bias)
        init_size: float for embeddings' initialization scale
        entity: torch.nn.Embedding with entity embeddings
        rel: torch.nn.Embedding with relation embeddings
        bh: torch.nn.Embedding with head entity bias embeddings
        bt: torch.nn.Embedding with tail entity bias embeddings
    """

    def __init__(self, sizes, rank, dropout, gamma, data_type, bias, init_size):
        """Initialize KGModel."""
        super(KGModel, self).__init__()
        if data_type == 'double':
            self.data_type = torch.double
        else:
            self.data_type = torch.float
        self.sizes = sizes
        self.rank = rank
        self.dropout = dropout
        self.bias = bias
        self.init_size = init_size
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
        self.entity = nn.Embedding(sizes[0], rank)
        self.rel = nn.Embedding(sizes[1], rank)
        self.bh = nn.Embedding(sizes[0], 1)
        self.bh.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)
        self.bt = nn.Embedding(sizes[0], 1)
        self.bt.weight.data = torch.zeros((sizes[0], 1), dtype=self.data_type)

    @abstractmethod
    def get_queries(self, queries):
        """Compute embedding and biases of queries.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
             lhs_e: torch.Tensor with queries' embeddings (embedding of head entities and relations)
             lhs_biases: torch.Tensor with head entities' biases
        """
        pass

    @abstractmethod
    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
             rhs_e: torch.Tensor with targets' embeddings
                    if eval_mode=False returns embedding of tail entities (n_queries x rank)
                    else returns embedding of all possible entities in the KG dataset (n_entities x rank)
             rhs_biases: torch.Tensor with targets' biases
                         if eval_mode=False returns biases of tail entities (n_queries x 1)
                         else returns biases of all possible entities in the KG dataset (n_entities x 1)
        """
        pass

    @abstractmethod
    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space.

        Args:
            lhs_e: torch.Tensor with queries' embeddings
            rhs_e: torch.Tensor with targets' embeddings
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            scores: torch.Tensor with similarity scores of queries against targets
        """
        pass

    def score(self, lhs, rhs, eval_mode):
        """Scores queries against targets

        Args:
            lhs: Tuple[torch.Tensor, torch.Tensor] with queries' embeddings and head biases
                 returned by get_queries(queries)
            rhs: Tuple[torch.Tensor, torch.Tensor] with targets' embeddings and tail biases
                 returned by get_rhs(queries, eval_mode)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            score: torch.Tensor with scores of queries against targets
                   if eval_mode=True, returns scores against all possible tail entities, shape (n_queries x n_entities)
                   else returns scores for triples in batch (shape n_queries x 1)
        """
        lhs_e, lhs_biases = lhs
        rhs_e, rhs_biases = rhs
        score = self.similarity_score(lhs_e, rhs_e, eval_mode)
        if self.bias == 'constant':
            return self.gamma.item() + score
        elif self.bias == 'learn':
            if eval_mode:
                return lhs_biases + rhs_biases.t() + score
            else:
                return lhs_biases + rhs_biases + score
        else:
            return score

    def mutiview_score(self):
        pass

    def get_factors(self, queries):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        rhs_e = self.entity(queries[:, 2])
        return head_e, rel_e, rhs_e
    
    def get_factors_without_euluc(self, queries, tensor_index):
        """Computes factors for embeddings' regularization.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor] with embeddings to regularize
        """
        head_e = self.entity(queries[:, 0])
        rel_e = self.rel(queries[:, 1])
        
        all_e_index = queries[:, 2]
        mask = torch.isin(all_e_index, tensor_index)
        other_e_index = all_e_index[mask]
        rhs_e = self.entity(other_e_index)
        return head_e, rel_e, rhs_e
    
    def get_euluc_factors(self, tensor_index):
        
        euluc_e = self.entity(tensor_index)
        return euluc_e
    
    def get_other_factors(self, tensor_index):
        # 获取euluc之外的其他embedding，tensor_index是euluc的index
        other_e = self.entity(tensor_index)
        return other_e

    def forward(self, queries, tensor_index=None, eval_mode=False):
        """KGModel forward pass.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            eval_mode: boolean, true for evaluation, false for training
        Returns:
            predictions: torch.Tensor with triples' scores
                         shape is (n_queries x 1) if eval_mode is false
                         else (n_queries x n_entities)
            factors: embeddings to regularize
        """
        # get embeddings and similarity scores
        lhs_e, lhs_biases = self.get_queries(queries)
        # queries = F.dropout(queries, self.dropout, training=self.training)
        rhs_e, rhs_biases = self.get_rhs(queries, eval_mode)
        # candidates = F.dropout(candidates, self.dropout, training=self.training)
        predictions = self.score((lhs_e, lhs_biases), (rhs_e, rhs_biases), eval_mode)

        # get factors for regularization
        if tensor_index is None:
            factors = self.get_factors(queries)
        else:
            # factors = self.get_factors_without_euluc(queries, tensor_index)
            factors = self.get_euluc_factors(tensor_index)
        return predictions, factors
    
    def get_ranking(self, queries, filters, batch_size=500):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            candidates = self.get_rhs(queries, eval_mode=True) # all entities embedding
                
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False) # label embedding

                scores = self.score(q, candidates, eval_mode=True) # all entities score
                targets = self.score(q, rhs, eval_mode=False) # label score

                # set filtered and true scores to -1e6 to be ignored 
                for i, query in enumerate(these_queries):
                    filter_out = [queries[b_begin + i, 2].item()]
                    # 使用idx2eulucclass过滤fliter_out
                    scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks
    
    def get_euluc_ranking(self, queries, filters, idx2eulucclass, batch_size=500):
        """Compute filtered ranking of correct entity for evaluation.

        Args:
            queries: torch.LongTensor with query triples (head, relation, tail)
            filters: filters[(head, relation)] gives entities to ignore (filtered setting)
            batch_size: int for evaluation batch size

        Returns:
            ranks: torch.Tensor with ranks or correct entities
        """
        ranks = torch.ones(len(queries))
        with torch.no_grad():
            b_begin = 0
            # candidates = self.get_rhs(queries, eval_mode=True) # all entities embedding
            eulucclass_idx = torch.LongTensor(list(idx2eulucclass.keys())).cuda()
            euluc_candidates = self.get_euluc_rhs(eulucclass_idx)
            
            # make idx2zeroindex
            idx_list = list(idx2eulucclass.keys())
            idx2zeroindex = {}
            for i in range(len(idx_list)):
                idx2zeroindex[idx_list[i]] = i
                
            while b_begin < len(queries):
                these_queries = queries[b_begin:b_begin + batch_size].cuda()

                q = self.get_queries(these_queries)
                rhs = self.get_rhs(these_queries, eval_mode=False) # label embedding

                # scores = self.score(q, candidates, eval_mode=True) # all entities score
                euluc_scores = self.score(q, euluc_candidates, eval_mode=True) # all entities score
                targets = self.score(q, rhs, eval_mode=False) # label score

                # set filtered and true scores to -1e6 to be ignored 
                for i, query in enumerate(these_queries):
                    # filter_out = filters[(query[0].item(), query[1].item())]
                    filter_out = [idx2zeroindex[queries[b_begin + i, 2].item()]]
                    # 使用idx2eulucclass过滤fliter_out
                    euluc_scores[i, torch.LongTensor(filter_out)] = -1e6
                ranks[b_begin:b_begin + batch_size] += torch.sum(
                    (euluc_scores >= targets).float(), dim=1
                ).cpu()
                b_begin += batch_size
        return ranks

    def compute_euluc_metrics(self, examples, filters, idx2eulucclass, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        q = examples.clone()

        ranks = self.get_euluc_ranking(q, filters['rhs'], idx2eulucclass, batch_size=batch_size)
        mean_rank = torch.mean(ranks).item()
        mean_reciprocal_rank = torch.mean(1. / ranks).item()
        hits_at = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 10)
        ))))
        metrics = {'MR': mean_rank, 'MRR': mean_reciprocal_rank, 'hits@[1,3,10]': hits_at}
        return metrics
    
    def compute_metrics(self, examples, filters, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores

        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        q = examples.clone()

        ranks = self.get_ranking(q, filters['rhs'], batch_size=batch_size)
        mean_rank = torch.mean(ranks).item()
        mean_reciprocal_rank = torch.mean(1. / ranks).item()
        hits_at = torch.FloatTensor((list(map(
            lambda x: torch.mean((ranks <= x).float()).item(),
            (1, 3, 10)
        ))))
        metrics = {'MR': mean_rank, 'MRR': mean_reciprocal_rank, 'hits@[1,3,10]': hits_at}
        return metrics
    def get_predict_results(self, examples, idx2eulucclass, batch_size=500):
        """Compute ranking-based evaluation metrics.
    
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores
            
        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        queries = examples.clone()
        results = np.zeros(len(queries))
        with torch.no_grad():
            b_begin = 0
            # candidates = self.get_rhs(queries, eval_mode=True) # all entities embedding
            eulucclass_idx = torch.LongTensor(list(idx2eulucclass.keys())).cuda()
            euluc_candidates = self.get_euluc_rhs(eulucclass_idx)
            
            results_score = np.zeros((len(queries), len(idx2eulucclass)))
            
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
                results_score[b_begin:b_begin + batch_size] = np.array([euluc_scores[i].cpu().numpy() for i in range(len(these_queries))])
                b_begin += batch_size
        np_query = queries.cpu().numpy()
        results_index = np.array([zeroindex2idx[idx] for idx in zeroindex2idx.keys()])
        # 拼接np_query和results
        results = np.hstack((np_query, results.reshape(-1, 1), results_score))
        return results, results_index
    
    def get_hit1_results(self, examples, filters, batch_size=500):
        """Compute ranking-based evaluation metrics.
        
        Args:
            examples: torch.LongTensor of size n_examples x 3 containing triples' indices
            filters: Dict with entities to skip per query for evaluation in the filtered setting
            batch_size: integer for batch size to use to compute scores
            
        Returns:
            Evaluation metrics (mean rank, mean reciprocical rank and hits)
        """
        mean_rank = {}
        mean_reciprocal_rank = {}
        hits_at = {}

        for m in ["rhs", "lhs"]:
            q = examples.clone()
            if m == "lhs":
                tmp = torch.clone(q[:, 0])
                q[:, 0] = q[:, 2]
                q[:, 2] = tmp
                q[:, 1] += self.sizes[1] // 2
            ranks = self.get_ranking(q, filters[m], batch_size=batch_size)
            mean_rank[m] = torch.mean(ranks).item()
            mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
            hits_at[m] = torch.FloatTensor((list(map(
                lambda x: torch.mean((ranks <= x).float()).item(),
                (1, 3, 10)
            ))))

        return mean_rank, mean_reciprocal_rank, hits_at
    
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
                all_embeddings[b_begin:b_begin + batch_size] = q[0].cpu()
                b_begin += batch_size

        return results, all_embeddings
            
