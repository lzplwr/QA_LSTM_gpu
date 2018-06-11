# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 17:19:53 2018

@author: hasee
"""

from modules.main import Chatter
from modules.nn import QA_LSTM_net
from moduels.util import remove
import requests
import json
import operator
import os
import readline

bot = Chatter()

net = QA_LSTM_net()
net.restore()
net.restore_embed()
net.set_eval()

code_path = os.path.realpath(__file__)
dir_path = os.path.dirname(code_path)
file_path = os.path.join(dir_path,'log','dialog_record.txt')

while True:
    q = input(':: ')
    if q == 'exit' or q == 'stop' or q == 'quit' or q == 'q':
        break
    else:
        # 如果用户输入空字符串，不作回复
        # 如果用户输入有效字符串，首先通过 bot(闲聊问答逻辑)搜索回答
        # 如果闲聊逻辑返回空回答，则同时搜索语料库和调用搜索引擎，并通过 net(神经网络模型)计算各个回答的匹配度
        # 最后返回匹配度最高的回答
        if not q:
            continue
        answer = bot.search_answer(q)
        if not answer:
            answer_cand = {}
            # 获取语料库中匹配度最高的答案
            sub_answer, sub_score = net.forward(q)
            sub_answer,has_removed = remove(sub_answer)
            if has_removed:
                sub_score = 0
            answer_cand[sub_answer] = sub_score
            # 获取搜索引擎返回的答案，并计算它们的匹配度
            '''
            response = requests.get('http://111.230.235.183:5000/qaByBd/' + q)
            response_dict = json.loads(response.text)
            if type(response_dict['data']) == str:
                sub_answer = response_dict['data'].strip()
                sub_score = net.cal_confidence(q, sub_answer)
                sub_answer, has_removed = remove(sub_answer)
                if has_removed:
                    sub_score = 0
                answer_cand[sub_r_answer] = sub_score
            else:
                for sub_dict in response_dict['data']:
                    for key in sub_dict:
                        sub_answer = sub_dict[key].strip()
                        sub_score = net.cal_confidence(q, sub_answer)
                        answer_cand[sub_answer] = sub_score
            '''
            sorted_answer = sorted(answer_cand.items(), key = operator.itemgetter(1), reverse = True)
            answer = sorted_answer[0][0]
        print('>>', answer)
        # 每次的问答都将存在在Log文件夹下的 dialog_record.txt 中
        with open(file_path,'a') as f:
            f.write('问：'+ q + '\n')
            f.write('答：' + answer + '\n')
            f.write('\n')