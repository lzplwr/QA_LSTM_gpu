# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 14:55:55 2018

@author: hasee
"""

import jieba

# segment是我们对句子的分词方法。这里我们采用字向量的方式输入
# 所以用list函数将句子逐字切割

def segment(u):
    return list(u)

# extract_gram 用于提取问句中的关键词，便于从语料库中快速检索相关问句
# 这里我们调用 jieba里的lcut_for_search方法，获取可能的关键词

def extract_gram(sent):
    gram_set = set(jieba.lcut_for_search(sent))
    return gram_set
