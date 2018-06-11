# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:24:46 2018

@author: hasee
"""

from modules.segment import extract_gram
import numpy as np
import re
import xlrd
import os

code_path = os.path.realpath(__file__)
dir_path = os.path.dirname(os.path.dirname(code_path))

# 读取语料库的函数 read_table

def read_table(file_name):
    file_path = os.path.join(dir_path, 'data', file_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheets()[0]
    questions = table.col_values(0)
    answers = table.col_values(1)
    data_pairs = [ (questions[i],answers[i]) for i in range(len(questions)) ]
    data_pairs = [ (str(q).strip(),str(a).strip()) for (q,a) in data_pairs if q and a ]
    return data_pairs

# 读取验证集和测试集的 read_test_table

def read_test_table(file_name):
    file_path = os.path.join(dir_path, 'data', file_name)
    data = xlrd.open_workbook(file_path)
    table = data.sheets()[0]
    questions = table.col_values(0)
    pos_answers = table.col_values(1)
    neg_answers = table.col_values(2)
    data_pairs = [ (questions[i],pos_answers[i],neg_answers[i]) for i in range(len(questions)) ]
    data_pairs = [ (str(q).strip(),str(p).strip(),str(n).strip()) for (q,p,n) in data_pairs if q and p and n ]
    return data_pairs

def read_answer():
    data = read_table('corpus.xlsx')
    answers = [ a for (q,a) in data ]
    return answers

def read_question():
    data = read_table('corpus.xlsx')
    questions = [ q for (q,a) in data ]
    return questions

# 每次使用新的训练集时，都要用 keyword_cal 函数重新计算关键词和它们的idf值

def keyword_cal():
    questions = read_question()
    gram_set = set()
    for q in questions:
        gram_set.update(extract_gram(q))
    idf_dict = {}
    gram2ques = {}
    for gram in gram_set:
        for index,q in enumerate(questions):
            if gram in q:
                idf_dict[gram] = idf_dict.get(gram, 0) + 1
                ques_list = gram2ques.get(gram, [])
                ques_list.append(index)
                gram2ques[gram] = ques_list
    idf_dict = { gram:np.log((len(questions)-idf_dict[gram]+0.5)/(idf_dict[gram]+0.5)) for gram in idf_dict }
    return idf_dict, gram2ques

# 移除低质量答案

re_words = ['头像','客服电话','客户经理','中国平安平安人寿']
punctuations = ['。','；','？','！']

def sentence_split(utterance):
    sents = []
    last_id = 0
    for i,c in enumerate(utterance):
        if c in punctuations or i == len(utterance) -1:
            sent = utterance[last_id:i+1]
            sents.append(sent)
            last_id = i+1
    return sents

def remove(utterance):
    sents = sentence_split(utterance)
    number = r'1' + r'\d' * 10
    number = re.compile(number)
    new_utterance = []
    for sent in sents:
        remove_flag = False
        result_list = number.findall(sent)
        if len(result_list):
            remove_flag = True
        for key in re_words:
            if key in sent:
                remove_flag = True
        if not remove_flag:
            new_utterance.append(sent)
    new_utterance = ''.join(new_utterance)
    if new_utterance != utterance:
        has_removed = True
    else:
        has_removed = False
    return new_utterance, has_removed