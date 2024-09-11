"""Dataset class for loading and processing UrbanKG datasets."""

import os
import pickle as pkl

import numpy as np
import torch

relation_types = {
    "vec": ['POI_In_Unit', 'Unit_Overlap_OSM', 'Unit_Adjacent_to_Unit', 'Fine_Class_Similar_to_EULUC_Class', 'Block_Adjacent_to_Block', 'Unit_In_Block', 'Unit_Overlap_Cell', 'Unit_Overlap_Area'],
    "seman": ['OSM_Class_Similar_to_EULUC_Class', 'Middle_Class_Belong_to_Coarse_Class', 'Fine_Class_Belong_to_Middle_Class', 'Cell_Class_Similar_to_EULUC_Class'],
    "vec_seman": ['Unit_Has_EULUC_Class', 'POI_Has_Fine_Class', 'Area_Has_Fine_Class', 'Cell_Has_Cell_Class', 'OSM_Has_OSM_Class']
}

class KGDataset(object):
    """Knowledge Graph dataset class."""

    def __init__(self, data_path, debug):
        """Creates KG dataset object for data loading.

        Args:
             data_path: Path to directory containing train/valid/test pickle files produced by process.py
             debug: boolean indicating whether to use debug mode or not
             if true, the dataset will only contain 1000 examples for debugging.
        """
        self.data_path = data_path
        self.debug = debug
        self.data = {}
        for split in ["train", "predict", "valid", "test"]:
            file_path = os.path.join(self.data_path, split + ".pickle")
            with open(file_path, "rb") as in_file:
                self.data[split] = pkl.load(in_file)
        
        self.entity2idx = {}
        self.relation2idx = {}
        entity2idx_file = open(os.path.join(self.data_path, "entity2id.txt"), "r")
        relation2idx_file = open(os.path.join(self.data_path, "relation2id.txt"), "r")
        with entity2idx_file as lines:
            for line in lines:
                entity, idx = line.strip().split("\t")
                self.entity2idx[entity] = int(idx)
        self.idx2entity = {v: k for k, v in self.entity2idx.items()}
        with relation2idx_file as lines:
            for line in lines:
                relation, idx = line.strip().split("\t")
                self.relation2idx[relation] = int(idx)
        self.idx2relation = {v: k for k, v in self.relation2idx.items()}
        self.eulucclass2idx = {}
        # 寻找entity2idx中的eulucclass, 格式为"eulucclass/xxx idx"
        self.eulucclass2idx = {k: v for k, v in self.entity2idx.items() if k.startswith("eulucclass/")}
        self.idx2eulucclass = {v: k for k, v in self.eulucclass2idx.items()}
        
        filters_file = open(os.path.join(self.data_path, "to_skip.pickle"), "rb")
        self.to_skip = pkl.load(filters_file)
        filters_file.close()
        all_data = np.vstack((self.data["train"], self.data["valid"]))
        max_axis = np.max(all_data, axis=0)
        self.n_entities = int(max(max_axis[0], max_axis[2]) + 1)
        self.n_predicates = int(max_axis[1] + 1) * 2

    def get_examples(self, split, rel_idx=-1):
        """Get examples in a split.

        Args:
            split: String indicating the split to use (train/valid/test)
            rel_idx: integer for relation index to keep (-1 to keep all relation)

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data[split]
        if split == "train":
            copy = np.copy(examples)
            tmp = np.copy(copy[:, 0])
            copy[:, 0] = copy[:, 2]
            copy[:, 2] = tmp
            copy[:, 1] += self.n_predicates // 2
            examples = np.vstack((examples, copy))
        if rel_idx >= 0:
            examples = examples[examples[:, 1] == rel_idx]
        if self.debug:
            examples = examples[:1000]
        return torch.from_numpy(examples.astype("int64"))
    
    def get_euluc_examples(self):
        """Get examples in a split.

        Returns:
            examples: torch.LongTensor containing KG triples in a split
        """
        examples = self.data['train']
        euluc_examples = []
        for example in examples:
            if example[1] == self.relation2idx['Unit_Has_EULUC_Class']:
                euluc_examples.append(example)
        np_euluc_examples = np.array(euluc_examples)
        return torch.from_numpy(np_euluc_examples.astype("int64"))
        
    
    def get_entity2idx(self):
        """Return entity2idx dictionary."""
        return self.entity2idx
    
    def get_idx2entity(self):
        """Return idx2entity dictionary."""
        return self.idx2entity
    
    def get_relation2idx(self):
        """Return relation2idx dictionary."""
        return self.relation2idx
    
    def get_idx2relation(self):
        """Return idx2relation dictionary."""
        return self.idx2relation
    
    def get_eulucclass2idx(self):
        """Return eulucclass2idx dictionary."""
        return self.eulucclass2idx
    
    def get_idx2eulucclass(self):
        """Return idx2eulucclass dictionary."""
        return self.idx2eulucclass
    
    def get_relation_type_index(self):
        relation_type_index = {}
        for index in relation_types:
            relation_type_index[index] = [self.relation2idx[relation] for relation in relation_types[index]]
        return relation_type_index

    def get_filters(self, ):
        """Return filter dict to compute ranking metrics in the filtered setting."""
        return self.to_skip

    def get_shape(self):
        """Returns KG dataset shape."""
        return self.n_entities, self.n_predicates, self.n_entities
