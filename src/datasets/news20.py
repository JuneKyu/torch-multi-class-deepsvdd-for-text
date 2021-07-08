#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np

import nltk
from nltk import word_tokenize
from torch.utils.data import Subset, DataLoader
from torchnlp.datasets.dataset import Dataset
from sklearn.datasets import fetch_20newsgroups

from .embeddings import preprocess_with_s_bert
from .preprocessing import get_target_label_idx, divide_data_label
from .misc import clean_text

import config

classes = [ 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware','comp.windows.x',
            'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
            'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
            'misc.forsale',
            'talk.politics.misc', 'talk.politics.guns', 'talk.politics.mideast',
            'talk.religion.misc', 'alt.atheism', 'soc.religion.christian'
        ]



class News20_Dataset():
    def __init__(self, root:str, normal_class:list):

        self.root = root
        self.n_classes = 2  # 0: normal, 1: outlier
        #  self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 20))

        comp = [0, 1, 2, 3, 4]
        rec = [5, 6, 7, 8]
        sci = [9, 10, 11, 12]
        misc = [13]
        pol = [14, 15, 16]
        rel = [17, 18, 19]
        total = comp + rec + sci + misc + pol + rel
        scenario_classes = (comp, rec, sci, misc, pol, rel)
        total.sort()

        normal_scenario = scenario_classes[normal_class[0]]
        self.normal_classes = []
        for normal in normal_scenario:
            self.normal_classes.append(total.index(normal))

        for i in normal_class:
            self.outlier_classes.remove(i)


        train_set = MyNews20(root=self.root, train=True, normal_classes=self.normal_classes)
        train_idx_normal = get_target_label_idx(train_set.ad_train_labels, self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)
        self.test_set = MyNews20(root=self.root, train=False, normal_classes=self.normal_classes)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_test=False, num_workers: int = 0) -> (
            DataLoader, DataLoader):
        train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
        test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=num_workers)
        return train_loader, test_loader


class MyNews20(Dataset):
    def __init__(self, root:str, train:bool, normal_classes:list):
        super(Dataset).__init__()

        self.normal_classes = normal_classes
        self.train_data, self.test_data = self.news20_dataset(root)
        self.train = train
        which_embedding = config.embedding
        assert which_embedding in config.implemented_nlp_embeddings

        print("embedding with {} embedding".format(which_embedding))


        if which_embedding == 'avg_glove':
            print("not implemented yet")
            #  self.train_x, self.train_y, self.test_x, self.test_y, self.train_text, self.test_text = \
            #      preprocess_with_avg_Glove(self.train, self.test)
        elif which_embedding == 'avg_bert':
            print("not implemented yet")
        #  download from nltk
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


    def news20_dataset(self, directory:str):
        clean_txt = True
        train = True
        test = True
        
        if directory not in nltk.data.path:
            nltk.data.path.append(directory)

        dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

        ret = []
        splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]
        for split_set in splits:
            dataset = fetch_20newsgroups(data_home=directory, subset=split_set, remove=('headers', 'footers', 'quotes'))
            examples = []

            for id in range(len(dataset.data)):
                if clean_txt:
                    text = clean_text(dataset.data[id])
                else:
                    text = ' '.join(word_tokenize(dataset.data[id]))
                label = dataset.target_names[int(dataset.target[id])]
                #  label = dataset.target[id]
                #  if label in self.normal_classes:
                #      label = 0
                #  else:
                #      label = 1
                
                if text:
                    examples.append({
                        'text': text,
                        'label': label
                    })
                    
            ret.append(Dataset(examples))

        ret_sentences = []
        ret_labels = []

        for ret_ in ret:

            sentence = []
            label = []

            for i, label_ in enumerate(ret_['label']):

                label_string = label_
                if label_string in classes:
                    label.append(classes.index(label_string))
                    sentence.append(ret_['text'][i])

            ret_sentences.append(sentence)
            ret_labels.append(label)

        train = pd.DataFrame({
            'sentence': ret_sentences[0],
            'label': ret_labels[0]
        })

        test = pd.DataFrame({'sentence': ret_sentences[1], 'label': ret_labels[1]})

        return train, test
