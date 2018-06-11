# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:23:33 2018

@author: hasee
"""

from modules.levenshtein import dalev
import modules.util as util

# 这是闲聊逻辑的定义，调用现成的字符串编辑距离函数，计算给定的问句和闲聊语料的问句的相似度

class Chatter():
    
    def __init__(self):
        self.qa_pairs = util.read_table('corpus_hi.xlsx')
    
    def search_answer(self, question):
        # 我们返回问候语料中最相似问句的答案。这里我们设置了一个阈值，当低于这个阈值时，就不返回答案
        best_answer,max_score = None,0.6
        for q,a in self.qa_pairs:
            score = self.similarity(q,question)
            if score > max_score:
                best_answer = a
                max_score = score
        return best_answer
    
    # 句子的相似度不仅与编辑距离有关，也和自身的长度有关。所以，这里用句子长度归一化字符串编辑距离，作为句子相似度
    
    def similarity(self, str1, str2):
        distance = dalev(str1, str2)
        score = 1 - 1.0 * distance/len(str1)
        return score