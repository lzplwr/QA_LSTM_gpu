# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 14:34:07 2018

@author: hasee
"""

from modules.nn import QA_LSTM_net
import modules.util as util
import random
import sys

# 定义训练的迭代次数 epochs，以及每个minibatch的包含的样本数量 batch_size

epochs = 50
batch_size = 200

class Trainer():
    
    def __init__(self):
        self.tr_dataset = util.read_table('train.xlsx')
        self.dev_dataset = util.read_test_table('dev.xlsx')
        self.ts_dataset = util.read_test_table('test.xlsx')
        self.net = QA_LSTM_net()
        self.max_acc = 0
    
    def train(self):
        print('\n:: training started\n')
        for j in range(epochs):
            # 每次迭代，都打乱训练集，并按批次训练数据，最后显示训练集的loss和验证集的正确率
            total_loss = 0.
            random.shuffle(self.tr_dataset)
            batch_num = int(len(self.tr_dataset)/batch_size)
            for i in range(batch_num):
                loss = self.net.train(self.tr_dataset[i*batch_size:(i+1)*batch_size])
                sys.stdout.write('\r{}.[{}/{}]'.format(j+1, i+1, batch_num))
                total_loss += loss
            print('\n\n:: {}.tr loss {}'.format(j+1, total_loss/batch_num))
            accuracy = self.evaluate(self.dev_dataset)
            print(':: {}.dev accuracy {}\n'.format(j+1, accuracy))
            # 比较这次迭代在验证集上正确率是否高于历史最高正确率，如果是，则保存网络
            if accuracy > self.max_acc:
                self.net.save()
                self.max_acc = accuracy
    
    def evaluate(self, dataset):
        self.net.set_eval()
        batch_num = int(len(dataset)/batch_size)
        correct_num = 0.
        for i in range(batch_num):
            correct_num += self.net.judge(dataset[i*batch_size:(i+1)*batch_size])
        self.net.set_train()
        return correct_num/(batch_num * batch_size)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
    acc = trainer.evaluate(trainer.ts_dataset)
    print('\n:: The test accuracy is {}'.format(acc))
    trainer.net.save_embed()