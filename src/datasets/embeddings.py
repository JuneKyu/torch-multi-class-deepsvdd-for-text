#!/usr/bin/env python
# -*- coding: utf-8 -*-


from sentence_transformers import SentenceTransformer

def sentence_bert_embed(sentences):
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentence_embeddings = model.encode(sentences)
    return sentence_embeddings, sentences.tolist()


def preprocess_with_s_bert(train, test):
    train_x, train_text = sentence_bert_embed(train.sentence.values)
    train_y = train.label.values
    test_x, test_text = sentence_bert_embed(test.sentence.values)
    test_y = test.label.values
    return train_x, train_y, test_x, test_y, train_text, test_text
