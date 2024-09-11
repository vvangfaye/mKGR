"""Knowledge Graph embedding model optimizer."""
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import nn


class KGOptimizer(object):
    """Knowledge Graph embedding model optimizer.

    KGOptimizers performs loss computations with negative sampling and gradient descent steps.

    Attributes:
        model: models.base.KGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
        neg_sample_size: An integer for the number of negative samples
        double_neg: A boolean (True to sample both head and tail entities)
    """

    def __init__(
            self, model, regularizer, optimizer, batch_size, neg_sample_size, double_neg, verbose=True):
        """Inits KGOptimizer."""
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.n_entities = model.sizes[0]

    def reduce_lr(self, factor=0.8):
        """Reduce learning rate.

        Args:
            factor: float for the learning rate decay
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

    def get_neg_samples(self, input_batch):
        """Sample negative examples.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            negative_batch: torch.Tensor of shape (neg_sample_size x 3) with negative examples
        """
        negative_batch = input_batch.repeat(self.neg_sample_size, 1)
        batch_size = input_batch.shape[0]
        negsamples = torch.Tensor(np.random.randint(
            self.n_entities,
            size=batch_size * self.neg_sample_size)
        ).to(input_batch.dtype)
        negative_batch[:, 2] = negsamples
        if self.double_neg:
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=batch_size * self.neg_sample_size)
            ).to(input_batch.dtype)
            negative_batch[:, 0] = negsamples
        return negative_batch

    def neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss with negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

        Returns:
            loss: torch.Tensor with negative sampling embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        # positive samples
        positive_score, factors = self.model(input_batch)
        positive_score = F.logsigmoid(positive_score)

        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        return loss, factors

    def no_neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss without negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        predictions, factors = self.model(input_batch, eval_mode=True)
        truth = input_batch[:, 2]
        log_prob = F.logsigmoid(-predictions)
        idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
        pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
        log_prob[idx, truth] += pos_scores
        loss = - log_prob.mean()
        loss += self.regularizer.forward(factors)
        return loss, factors

    def calculate_loss(self, input_batch):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth = input_batch[:, 2]
            loss = self.loss_fn(predictions, truth)
            # loss, factors = self.no_neg_sampling_loss(input_batch)

        # regularization loss
        loss += self.regularizer.forward(factors)
        return loss

    def calculate_valid_loss(self, examples):
        """Compute KG embedding loss over validation examples.

        Args:
            examples: torch.LongTensor of shape (N_valid x 3) with validation triples

        Returns:
            loss: torch.Tensor with loss averaged over all validation examples
        """
        b_begin = 0
        loss = 0.0
        counter = 0
        with torch.no_grad():
            while b_begin < examples.shape[0]:
                input_batch = examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                b_begin += self.batch_size
                loss += self.calculate_loss(input_batch)
                counter += 1
        loss /= counter
        return loss

    def epoch(self, examples):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: torch.Tensor with loss averaged over all training examples
        """
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            total_loss = 0.0
            counter = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()

                # gradient step
                l = self.calculate_loss(input_batch)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                b_begin += self.batch_size
                total_loss += l
                counter += 1
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.4f}')
        total_loss /= counter
        return total_loss    

class KGOptimizerEuluc(object):
    """Knowledge Graph embedding model optimizer.

    KGOptimizersEuluc performs loss computations with negative sampling and gradient descent steps.

    Attributes:
        model: models.base.KGModel
        regularizer: regularizers.Regularizer
        optimizer: torch.optim.Optimizer
        batch_size: An integer for the training batch size
        neg_sample_size: An integer for the number of negative samples
        double_neg: A boolean (True to sample both head and tail entities)
    """

    def __init__(
            self, model, regularizer, optimizer, batch_size, neg_sample_size, euluc_batch_size, euluc_neg_sample_size, double_neg, idx2eulucclass, verbose=True):
        """Inits KGOptimizerEuluc."""
        self.model = model
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.verbose = verbose
        self.double_neg = double_neg
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.neg_sample_size = neg_sample_size
        self.gamma = 0.5
        
        self.euluc_batch_size = euluc_batch_size
        self.euluc_neg_sample_size = euluc_neg_sample_size
        self.n_entities = model.sizes[0]
        self.idx2eulucclass = idx2eulucclass

    def reduce_lr(self, factor=0.8):
        """Reduce learning rate.

        Args:
            factor: float for the learning rate decay
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= factor

    def neg_atation(self, euluc_positive_score, euluc_negative_score):
        """Compute KG embedding loss with negative sampling.

        Args:
            euluc_positive_score: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.
            euluc_negative_score: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.
        """
        attentions = torch.chunk(euluc_negative_score, self.euluc_neg_sample_size, 0)
        attentions = torch.cat(attentions, 1)
        # sum
        attentions = torch.nn.functional.softmax(attentions, dim=1) * attentions
        attentions = torch.sum(attentions, 1)
        attentions = 1 / - attentions

        attentions = attentions.unsqueeze(1)
        euluc_positive_score = euluc_positive_score * attentions
        # attentions = attentions.unsqueeze(1)
        # euluc_positive_score = euluc_positive_score - attentions
        return euluc_positive_score

    def normal_loss(self, positive_score, negative_score):
        negative_score_chunk = torch.chunk(negative_score, self.neg_sample_size, 0)
        negative_score_chunk = torch.cat(negative_score_chunk, 1)
        # sum
        negative_score = torch.nn.functional.softmax(negative_score_chunk, dim=1) * negative_score_chunk
        negative_score = torch.sum(negative_score, 1)
        loss = self.gamma - positive_score + negative_score
        # 取绝对值
        loss = torch.abs(loss)
        return loss

    def euluc_loss(self, euluc_positive_score, euluc_negative_score):
        euluc_negative_score_chunk = torch.chunk(euluc_negative_score, self.euluc_neg_sample_size, 0)
        euluc_negative_score_chunk = torch.cat(euluc_negative_score_chunk, 1)
        # sum
        euluc_negative_score = torch.nn.functional.softmax(euluc_negative_score_chunk, dim=1) * euluc_negative_score_chunk
        euluc_negative_score = torch.sum(euluc_negative_score, 1)
        loss = self.gamma - euluc_positive_score + euluc_negative_score
        # 取绝对值
        loss = torch.abs(loss)
        return loss


    def neg_atation_CANS(self, euluc_positive_score, euluc_negative_score):
        """Compute KG embedding loss with negative sampling.

        Args:
            euluc_positive_score: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.
            euluc_negative_score: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.
        """
        attentions = torch.chunk(euluc_negative_score, self.euluc_neg_sample_size, 0)
        attentions = torch.cat(attentions, 1)
        # sum
        attentions = torch.nn.functional.softmax(attentions, dim=0) * attentions
        attentions = torch.sum(attentions, 1)
        attentions = attentions.unsqueeze(1)
        attentions = - (self.gamma - euluc_positive_score + attentions)
        attentions = attentions.repeat(1, euluc_positive_score.shape[1])
        euluc_positive_score = euluc_positive_score * attentions
        # attentions = attentions.unsqueeze(1)
        # euluc_positive_score = euluc_positive_score - attentions
        return euluc_positive_score

    def get_neg_samples(self, input_batch):
        """Sample negative examples.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            negative_batch: torch.Tensor of shape (neg_sample_size x 3) with negative examples
        """
        # negative_batch = input_batch.repeat(self.neg_sample_size, 1)
        negative_batch = input_batch.repeat_interleave(self.neg_sample_size, 0)
        batch_size = input_batch.shape[0]
        negsamples = torch.Tensor(np.random.randint(
            self.n_entities,
            size=batch_size * self.neg_sample_size)
        ).to(input_batch.dtype)
        negative_batch[:, 2] = negsamples
        if self.double_neg:
            negsamples = torch.Tensor(np.random.randint(
                self.n_entities,
                size=batch_size * self.neg_sample_size)
            ).to(input_batch.dtype)
            negative_batch[:, 0] = negsamples
        return negative_batch
    
    def get_euluc_neg_samples(self, input_euluc_batch):
        """Sample euluc negative examples.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            negative_batch: torch.Tensor of shape (neg_sample_size x 3) with negative examples
        """
        # negative_euluc_batch = input_euluc_batch.repeat(self.euluc_neg_sample_size, 1)
        negative_euluc_batch = input_euluc_batch.repeat_interleave(self.euluc_neg_sample_size, 0)
        batch_size = input_euluc_batch.shape[0]
        euluc_idx = self.idx2eulucclass.keys()
        negsamples = torch.Tensor(np.random.choice(
            list(euluc_idx),
            size=batch_size * self.euluc_neg_sample_size)
        ).to(input_euluc_batch.dtype)
        negative_euluc_batch[:, 2] = negsamples
        return negative_euluc_batch

    def neg_sampling_loss(self, input_batch, input_euluc_batch):
        """Compute KG embedding loss with negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

        Returns:
            loss: torch.Tensor with negative sampling embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        
        # tensor_index = torch.tensor(list(self.idx2eulucclass.keys())).cuda()
        # positive samples
        positive_score, factors = self.model(input_batch)
        positive_score = F.logsigmoid(positive_score)
        
        # positive euluc samples
        euluc_positive_score, _ = self.model(input_euluc_batch)
        # euluc_positive_score = F.logsigmoid(euluc_positive_score)

        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        
        # negative euluc samples
        euluc_neg_samples = self.get_euluc_neg_samples(input_euluc_batch)
        euluc_negative_score, _ = self.model(euluc_neg_samples)

        euluc_positive_score = F.logsigmoid(euluc_positive_score)
        euluc_negative_score = F.logsigmoid(-euluc_negative_score)

        euluc_positive_score = self.neg_atation(euluc_positive_score, euluc_negative_score)

        # euluc_positive_score = self.neg_atation_CANS(euluc_positive_score, euluc_negative_score)
        # normal_loss = self.normal_loss(positive_score, negative_score)
        # euluc_loss = self.euluc_loss(euluc_positive_score, euluc_negative_score)
        
        
        # loss = - (torch.cat([positive_score, negative_score, euluc_positive_score, euluc_negative_score], dim=0).mean() + self.regularizer.forward(euluc_factors))
        loss = - torch.cat([positive_score, negative_score, euluc_positive_score, euluc_negative_score], dim=0).mean()

        # loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        # loss = normal_loss.mean() + euluc_loss.mean()
        # loss += 1 / self.regularizer.forward(euluc_factors)
        
        # loss = - torch.cat([positive_score, negative_score], dim=0).mean()

        return loss, factors
    
    def neg_sampling_loss_no_euluc(self, input_batch):
        """Compute KG embedding loss with negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples.

        Returns:
            loss: torch.Tensor with negative sampling embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        # positive samples
        positive_score, factors = self.model(input_batch)
        positive_score = F.logsigmoid(positive_score)

        # negative samples 
        neg_samples = self.get_neg_samples(input_batch)
        negative_score, _ = self.model(neg_samples)
        negative_score = F.logsigmoid(-negative_score)
        
        loss = - torch.cat([positive_score, negative_score], dim=0).mean()
        # loss = - torch.cat([positive_score, euluc_positive_score, euluc_negative_score], dim=0).mean()
        return loss, factors
        
    def no_neg_sampling_loss(self, input_batch):
        """Compute KG embedding loss without negative sampling.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss
            factors: torch.Tensor with embeddings weights to regularize
        """
        predictions, factors = self.model(input_batch, eval_mode=True)
        truth = input_batch[:, 2]
        log_prob = F.logsigmoid(-predictions)
        idx = torch.arange(0, truth.shape[0], dtype=truth.dtype)
        pos_scores = F.logsigmoid(predictions[idx, truth]) - F.logsigmoid(-predictions[idx, truth])
        log_prob[idx, truth] += pos_scores
        loss = - log_prob.mean()
        loss += self.regularizer.forward(factors)
        return loss, factors

    def calculate_loss(self, input_batch, input_euluc_batch):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss(input_batch, input_euluc_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth = input_batch[:, 2]
            loss = self.loss_fn(predictions, truth)
            # loss, factors = self.no_neg_sampling_loss(input_batch)

        # regularization loss
        # loss += self.regularizer.forward(factors)
        return loss
    
    def calculate_loss_no_euluc(self, input_batch):
        """Compute KG embedding loss and regularization loss.

        Args:
            input_batch: torch.LongTensor of shape (batch_size x 3) with ground truth training triples

        Returns:
            loss: torch.Tensor with embedding loss and regularization loss
        """
        if self.neg_sample_size > 0:
            loss, factors = self.neg_sampling_loss_no_euluc(input_batch)
        else:
            predictions, factors = self.model(input_batch, eval_mode=True)
            truth = input_batch[:, 2]
            loss = self.loss_fn(predictions, truth)
            # loss, factors = self.no_neg_sampling_loss(input_batch)

        # regularization loss
        loss += self.regularizer.forward(factors)
        return loss
    
    def calculate_valid_loss(self, examples):
        """Compute KG embedding loss over validation examples.

        Args:
            examples: torch.LongTensor of shape (N_valid x 3) with validation triples

        Returns:
            loss: torch.Tensor with loss averaged over all validation examples
        """
        b_begin = 0
        loss = 0.0
        counter = 0
        with torch.no_grad():
            while b_begin < examples.shape[0]:
                input_batch = examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                b_begin += self.batch_size
                loss += self.calculate_loss(input_batch, input_batch)
                counter += 1
        loss /= counter
        return loss

    def epoch(self, examples, euluc_examples):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: torch.Tensor with loss averaged over all training examples
        """
        # if self.lwr_attention is None:
        #     self.lwr_attention = torch.ones((euluc_examples.shape[0], 1)).cuda()

        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        actual_euluc_examples = euluc_examples[torch.randperm(euluc_examples.shape[0]), :]

        # actual_euluc_examples_list = []
        # euluc_set = set()
        # for i in range(actual_euluc_examples.shape[0]):
        #     if actual_euluc_examples[i, 2].item() not in euluc_set:
        #         euluc_set.add(actual_euluc_examples[i, 2].item())
        # for i in euluc_set:
        #     actual_euluc_examples_list.append(actual_euluc_examples[actual_euluc_examples[:, 2] == i])
        
        # each_euluc_batch_num = self.euluc_batch_size // len(actual_euluc_examples_list)
            
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            # euluc_b_begin = list(np.zeros(len(actual_euluc_examples_list), dtype=int))
            euluc_b_begin = 0
            total_loss = 0.0
            counter = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()
                
                
                if euluc_b_begin + self.euluc_batch_size > euluc_examples.shape[0]:
                    input_euluc_batch = actual_euluc_examples[
                                euluc_b_begin:euluc_examples.shape[0]
                                ].cuda()
                    euluc_pointer = (euluc_b_begin, euluc_examples.shape[0])
                    euluc_b_begin = 0
                else:
                    input_euluc_batch = actual_euluc_examples[
                                    euluc_b_begin:euluc_b_begin + self.euluc_batch_size
                                    ].cuda()
                    euluc_pointer = (euluc_b_begin, euluc_b_begin + self.euluc_batch_size)
                    euluc_b_begin += self.euluc_batch_size
                # input_euluc_batch = []
                # for i in range(len(actual_euluc_examples_list)):
                #     if euluc_b_begin[i] + each_euluc_batch_num > actual_euluc_examples_list[i].shape[0]:
                #         input_euluc_batch.append(actual_euluc_examples_list[i][
                #                     euluc_b_begin[i]:actual_euluc_examples_list[i].shape[0]
                #                     ].cuda())
                #         euluc_b_begin[i] = 0
                #     else:
                #         input_euluc_batch.append(actual_euluc_examples_list[i][
                #                         euluc_b_begin[i]:euluc_b_begin[i] + each_euluc_batch_num
                #                         ].cuda())
                #     euluc_b_begin[i] += each_euluc_batch_num
                # input_euluc_batch = torch.cat(input_euluc_batch, dim=0)

                # gradient step
                l = self.calculate_loss(input_batch, input_euluc_batch)
                # l = self.calculate_loss_no_euluc(input_batch)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                b_begin += self.batch_size

                total_loss += l
                counter += 1
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.4f}')
        total_loss /= counter
        return total_loss
    
    def epoch_no_euluc(self, examples):
        """Runs one epoch of training KG embedding model.

        Args:
            examples: torch.LongTensor of shape (N_train x 3) with training triples

        Returns:
            loss: torch.Tensor with loss averaged over all training examples
        """
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
            
        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            
            total_loss = 0.0
            counter = 0
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                              b_begin:b_begin + self.batch_size
                              ].cuda()

                # gradient step
                l = self.calculate_loss_no_euluc(input_batch)
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                b_begin += self.batch_size
                # euluc_b_begin += self.euluc_batch_size
                total_loss += l
                counter += 1
                bar.update(input_batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.4f}')
        total_loss /= counter
        return total_loss
