#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

from torch.utils.data import Subset, DataLoader
from torchnlp.datasets.dataset import Dataset
from torchnlp.datasets import trec_dataset

from .embeddings import preprocess_with_s_bert
from .preprocessing import get_target_label_idx, divide_data_label
from .misc import clean_text

import config


classes = ['ABBR', # Abbreviation
           'DESC', # Description and abstract concepts
           'ENTY', # Entities
           'HUM', # Human beings
           'LOC', # Locations
           'NUM'] # Numeric values

class Trec_Dataset():
    def __init__(self, root:str, normal_class:list):

        self.root = root
        self.n_classes = 2
        self.outlier_classes = list(range(0, 6))
        self.normal_classes = normal_class
        for i in normal_class:
            self.outlier_classes.remove(i)

        train_set = MyTrec(root=self.root, train=True, normal_classes=self.normal_classes)
        train_idx_normal = get_target_label_idx(train_set.ad_train_labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)
        self.test_set = MyTrec(root=self.root, train=False, normal_classes=self.normal_classes)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, test_loader


class MyTrec(Dataset):
    def __init__(self, root:str, train:bool, normal_classes:list):
        super(Dataset).__init__()
        
        self.normal_classes = normal_classes
        self.train_data, self.test_data = self.trec_dataset(directory=root, clean_txt=True)
        self.train = train
        which_embedding = config.embedding
        assert which_embedding in config.implemented_nlp_embeddings

        if which_embedding == 'avg_glove':
            print("not implemented yet")
        elif which_embedding == 'avg_bert':
            print("not implemented yet")
        elif which_embedding == 's_bert':
            self.train_data, self.ad_train_labels, self.test_data, self.ad_test_labels, self.train_text, self.test_text = \
                preprocess_with_s_bert(self.train_data, self.test_data)
        elif which_embedding == 'avg_fasttext':
            print("not implemented yet")

        self.train_labels = []
        for train_label in self.ad_train_labels:
            if train_label in self.normal_classes:
                self.train_labels.append(0)
            else:
                self.train_labels.append(1)
        self.train_labels = np.array(self.train_labels)
        self.test_labels = []
        for test_label in self.ad_test_labels:
            if test_label in self.normal_classes:
                self.test_labels.append(0)
            else:
                self.test_labels.append(1)
        self.test_labels = np.array(self.test_labels)

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, target = self.train_data[index], self.train_labels[index]
        else:
            data, target = self.test_data[index], self.test_labels[index]

        return data, target, index


    def trec_dataset(self, directory:str, clean_txt:bool):
        train, test = trec_dataset(directory=directory, train=True, test=True)
        train_set = {'label':[], 'sentence':[]}
        for data in train:
            train_set['label'].append(classes.index(data['label']))
            if clean_txt:
                train_set['sentence'].append(clean_text(data['text'].lower()))
            else:
                train_set['sentence'].append(data['text'].lower())

        test_set = {'label':[], 'sentence':[]}
        for data in test:
            test_set['label'].append(classes.index(data['label']))
            if clean_txt:
                test_set['sentence'].append(clean_text(data['text'].lower()))
            else:
                test_set['sentence'].append(data['text'].lower())

        train = pd.DataFrame(train_set)
        test = pd.DataFrame(test_set)

        return train, test
