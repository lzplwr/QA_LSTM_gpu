# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:41:18 2018

@author: hasee
"""

import torch
from torch.autograd import Variable
from modules.segment import segment,extract_gram
import modules.util as util
from gensim.models.keyedvectors import KeyedVectors
import random
import numpy as np
from pickle import load,dump
import operator
import os

code_path = os.path.realpath(__file__)
dir_path = os.path.dirname(os.path.dirname(code_path))

nb_hidden = 256

max_ql = 50
max_al = 200
margin = 0.1
drop_rate = 0
num_layers = 2
clip_norm = 5
learning_rate = 0.001

min_idf = 4
candicate_num = 100

class RNN(torch.nn.Module):
    
    def __init__(self, nb_hidden, embed_size):
        super(RNN,self).__init__()
        self.rnn = torch.nn.LSTM(input_size = embed_size, hidden_size = nb_hidden, num_layers = \
                                 num_layers, batch_first = True, dropout = drop_rate, bidirectional = True)
        
    def forward(self,inx):
        self.rnn.flatten_parameters()
        lstm_out, _ = self.rnn(inx, None)
        return lstm_out

class QA_LSTM_net():
    
    def __init__(self):
        model_path = os.path.join(dir_path,'model','zi5_vec.model')
        self.wv = KeyedVectors.load(model_path)
        self.word_dim = np.shape(self.wv[self.wv.index2word])[1]
        self.rnn = RNN(nb_hidden,self.word_dim)
        self.rnn.cuda()
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr = learning_rate)
        #others
        self.answers = util.read_answer()
        #ready for evaluate
        self.questions = util.read_question()
        self.avg_ql = 0.0
        for q in self.questions:
            self.avg_ql += len(q)
        self.avg_ql = self.avg_ql/len(self.questions)
        file_path = os.path.join(dir_path,'model','idf_dict')
        with open(file_path, 'rb') as f:
            self.idf_dict = load(f)
        file_path = os.path.join(dir_path,'model','gram2ques')
        with open(file_path, 'rb') as f:
            self.gram2ques = load(f)
    
    def forward(self, question):
        a_embs,a_candicates = self.search_candicate(question)
        if len(a_candicates) == 0:
            return '请更具体地描述你的问题。', 0.
        q_embs = self.embed([question], max_ql)
        scores = torch.nn.functional.cosine_similarity(q_embs, a_embs)
        index = scores.cpu().data.numpy().argmax()
        score = scores.cpu().data.numpy()[index]
        return a_candicates[index],score
    
    def cal_confidence(self, question, answer):
        q_embs = self.embed([question], max_ql)
        a_embs = self.embed([answer], max_al)
        score = torch.nn.functional.cosine_similarity(q_embs, a_embs)
        return score.cpu().data.numpy()[0]
    
    def judge(self, batch):
        questions = [ q for (q,_,_) in batch ]
        q_embs = self.embed(questions, max_ql)
        pos_answers = [ p for (_,p,_) in batch ]
        p_embs = self.embed(pos_answers, max_al)
        positive_score = torch.nn.functional.cosine_similarity(q_embs, p_embs)
        neg_answers_id = [ n.split(' ') for (_,_,n) in batch ]
        neg_answers_id = [ [ int(i) for i in n ] for n in neg_answers_id ]
        neg_answers_id = np.array(neg_answers_id)
        negative_score = torch.zeros(neg_answers_id.shape)
        negative_score = Variable(negative_score).cuda()
        for i in range(100):
            neg_answers = [ self.answers[j] for j in neg_answers_id[:,i] ]
            n_embs = self.embed(neg_answers, max_al)
            sub_score = torch.nn.functional.cosine_similarity(q_embs, n_embs)
            sub_score = Variable(sub_score.data)
            negative_score[:,i] = sub_score
        negative_score, _ = torch.max(negative_score, dim = 1)
        acc =  positive_score > negative_score
        acc = acc.cpu().data.numpy()
        return acc.sum()
    
    def train(self, tr_batch):
        questions = [ q for (q,a) in tr_batch ]
        answers = [ a for (q,a) in tr_batch ]
        #question embed
        q_embs = self.embed(questions,max_ql)
        #positive answer embed
        positive_embs = self.embed(answers,max_al)
        #negative answer embed
        negative_answers = random.sample(self.answers,len(tr_batch))
        negative_embs = self.embed(negative_answers,max_al)
        #Margin loss
        positive_score = torch.nn.functional.cosine_similarity(q_embs, positive_embs)
        negative_score = torch.nn.functional.cosine_similarity(q_embs, negative_embs)
        loss = self.loss_func(positive_score, negative_score)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.rnn.parameters(), clip_norm)
        self.optimizer.step()
        return loss.cpu().data.numpy()[0]
    
    def save_embed(self):
        answer_embedding = np.zeros(shape = (len(self.answers), nb_hidden * 2))
        for i,a in enumerate(self.answers):
            answer_embedding[i] = self.embed([a], max_al).view(-1).cpu().data.numpy()
        self.answer_embedding = answer_embedding
        model_file = os.path.join(dir_path,'model','answer_embed.pkl')
        with open(model_file, 'wb') as f:
            dump(self.answer_embedding, f)
    
    def restore_embed(self):
        model_file = os.path.join(dir_path,'model','answer_embed.pkl')
        with open(model_file, 'rb') as f:
            self.answer_embedding = load(f)
    
    def embed(self, utterances, max_l):
        u_embs = np.zeros((len(utterances),max_l,self.word_dim))
        for i,u in enumerate(utterances):
            word_list = segment(u)
            word_list = [ word for word in word_list if word in self.wv ]
            word_list = word_list[:max_l]
            if not word_list:
                word_list = list('<空>')
            u_embs[i,:len(word_list)] = self.wv[word_list]
        u_embs = torch.FloatTensor(u_embs)
        u_embs = Variable(u_embs).cuda()
        u_embs = self.rnn(u_embs)
        max_pool = torch.nn.MaxPool2d(kernel_size = (max_l, 1), stride = 1)
        u_embs = max_pool(u_embs)
        u_embs = u_embs.view(-1, nb_hidden * 2)
        return u_embs
    
    def loss_func(self,positive_var,negative_var):
        margin_var = Variable(torch.ones(1)*margin).cuda()
        diff_var = margin_var - positive_var + negative_var
        loss = diff_var * (diff_var > 0).type(torch.cuda.FloatTensor)
        loss = loss.mean()
        return loss
    
    def search_candicate(self, question):
        grams = extract_gram(question)
        q_candicates = set()
        for gram in grams:
            if self.idf_dict.get(gram, -1) > min_idf:
                q_candicates.update(self.gram2ques[gram])
        q_candicates = { i:self.BM25(grams, self.questions[i]) for i in q_candicates }
        sorted_q = sorted(q_candicates.items(), key = operator.itemgetter(1), reverse = True)
        a_candicates = [ self.answers[index] for (index,r) in sorted_q ]
        a_candid = [ index for (index,r) in sorted_q ]
        a_candid = a_candid[:candicate_num]
        a_candid = np.array(a_candid)
        if len(a_candicates) == 0:
            return None, a_candicates
        a_embs = self.answer_embedding[a_candid,:]
        a_embs = Variable(torch.Tensor(a_embs)).cuda()
        return a_embs,a_candicates
    
    def BM25(self, q1_grams, q2):
        r = 0.0
        for gram in q1_grams:
            gram_count = q2.count(gram)
            r += self.idf_dict.get(gram, 0)*3*gram_count/(gram_count+0.5+1.5*len(q2)/self.avg_ql)
        return r
    
    def set_eval(self):
        self.rnn.eval()
    
    def set_train(self):
        self.rnn.train()
    
    def save(self):
        model_file = os.path.join(dir_path,'model','main_rnn.pkl')
        torch.save(self.rnn, model_file)
        print('\n:: saved to model/main_rnn.pkl \n')
    
    def restore(self):
        model_file = os.path.join(dir_path,'model','main_rnn.pkl')
        self.rnn = torch.load(model_file)
        print('\n:: restoring checkpoint from', model_file, '\n')