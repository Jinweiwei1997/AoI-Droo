#  #################################################################
#  This file contains the main DROO operations, including building DNN, 
#  Storing data sample, Training DNN, and generating quantized binary offloading decisions.

#  version 1.0 -- February 2020. Written based on Tensorflow 2 by Weijian Pan and 
#  Liang Huang (lianghuang AT zjut.edu.cn)
#  ###################################################################

from __future__ import print_function

import operator
import cProfile
import time

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from bisection import bisection

print(torch.__version__)


# DNN network for memory
class MemoryDNN:
    def __init__(
        self,
        net,
        learning_rate = 0.00005,
        training_interval=10,
        batch_size=100,
        memory_size=1000,
        output_graph=False
    ):

        self.net = net
        self.training_interval = training_interval      # learn every #training_interval
        self.lr = learning_rate
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1
        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
                nn.Linear(self.net[0], self.net[1]),
                nn.ReLU(),
                nn.Linear(self.net[1], self.net[2]),
                nn.ReLU(),
                nn.Linear(self.net[2], self.net[3]),
                nn.BatchNorm1d(self.net[3]),
                nn.Sigmoid(),


        )

    def remember(self, h,g,BEnergy,AoI, m):


        # replace the old memory with new memory

        idx = self.memory_counter % self.memory_size
        assert (m[np.argmax(m)] == 1)
        self.memory[idx, :] = np.hstack((h,g,BEnergy,AoI, m))
        self.memory_counter += 1


    def encode(self, h, g,BEnergy,AoI,m):
        # encoding the entry
        assert (m[np.argmax(m)] == 1)
        self.remember(h,g,BEnergy,AoI, m)
        # train the DNN every 10 step
#        if self.memory_counter> self.memory_size / 2 and self.memory_counter % self.training_interval == 0:
        if  self.memory_counter % self.training_interval == 0 :
            self.learn()

    def learn(self):

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = torch.Tensor(batch_memory[:, 0: self.net[0]])
        # print('==============',h_train.shape)
        m_train = torch.Tensor(batch_memory[:, self.net[0]:])
        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(h_train)
        loss = criterion(predict, m_train)

        loss.backward()
        optimizer.step()

        self.cost = loss.item()
        #assert(self.cost > 0)
        self.cost_his.append(self.cost)
        '''
        if(self.memory_counter > 30000 and (self.memory_counter-30000)%128==0):
            train_acc = 0
            for x in range(128):
                flat = 0
                train_correct = 0

                if torch.Tensor.argmax((predict[x, :]))==np.argmax(m_train[x, :]):

                    train_correct = 1
                    train_acc += train_correct
            print("准确率",train_acc/128)
'''
    def decode(self, h,g,BEnergy,AoI, k , mode = 'OP'):
        if mode =='OP':    #遍历所有的动作，因为设定的动作至多又一个1
           return self.allAction(k)
        if mode=='Choose': #选择动作
            return self.chooseAction1( h,g,BEnergy,AoI,k)
    def allAction(slef,k):
        m_list = []
        for i in range(k + 1):
            m_index = []
            for j in range(k + 1):
                if (i == j):
                    m_index.append(1)
                else:
                    m_index.append(0)
            m_list.append(m_index)
        return m_list
    def chooseAction(self, h,g,BEnergy,AoI,k):
        m_list=[]
        bisection_list=[]
        action_number=int(5)
        t0=time.time()
        self.model.eval()
        th=time.time()
        th0=th-t0
        temp = torch.Tensor([np.hstack((h, g, BEnergy, AoI))])
        t1=time.time()
        t1h=t1-th
        # print('===', temp.shape)
        predict = self.model(temp)
        predict = predict.detach().numpy()
        t2=time.time()
        t12=t2-t1
        # print('------',predict)
        list_in=[]
        list_in=np.argsort(-predict)
        # for i in range(k+1):
        #     max=0
        #     flat=0
        #     for j in range(k+1):
        #         if predict[0,j]>=max and (j not in list_in) :
        #             max=predict[0,j]
        #             flat=j
        #     list_in.append(flat)
        t3=time.time()
        t23=t3-t2
        flat_number=0
        for node_to_trans in list_in[0] :
            m_index=[]
            for i in range(k+1):
                if i==node_to_trans:
                    m_index.append(1)
                else:
                    m_index.append(0)
            B_bb=[x/1000 for x in BEnergy]
            t4=time.time()
            x=bisection(h / 10000, g / 10000, B_bb, AoI, m_index)
            t5=time.time()
            t45=t5-t4
            if x[0]> -1000:
                m_list.append(m_index)
                bisection_list.append(x)
                flat_number+=1
            if flat_number ==action_number:
                break
        t6=time.time()
        t63=t6-t3
        return m_list,bisection_list
    #cProfile.run('chooseAction()')
    def allAction(slef,k):
        m_list = []
        for i in range(k + 1):
            m_index = []
            for j in range(k + 1):
                if (i == j):
                    m_index.append(1)
                else:
                    m_index.append(0)
            m_list.append(m_index)
        return m_list

    def chooseAction1(self, h,g,BEnergy,AoI,k):
        m_list=[]
        bisection_list=[]
        action_number=int(5)
        t0=time.time()
        self.model.eval()
        temp = torch.Tensor([np.hstack((h, g, BEnergy, AoI))])
        t1=time.time()
        t01=t1-t0
        # print('===', temp.shape)
        predict = self.model(temp)
        predict = predict.detach().numpy()
        t2=time.time()
        t12=t2-t1
        # print('------',predict)
        # for i in range(k+1):
        #     max=0
        #     flat=0
        #     for j in range(k+1):
        #         if predict[0,j]>=max and (j not in list_in) :
        #             max=predict[0,j]
        #             flat=j
        #     list_in.append(flat)
        t3=time.time()
        t23=t3-t2
        flat_number=0
        list_in=[x for x in predict[0]]
        for j in range(len(predict[0])) :
            node_to_trans=np.argmax(list_in)
            m_index=[]
            for k in range(k+1):
                if k==node_to_trans:
                    m_index.append(1)
                else:
                    m_index.append(0)
            B_bb=[x/1000 for x in BEnergy]
            t4=time.time()
            x=bisection(h / 10000, g / 10000, B_bb, AoI, m_index)
            t5=time.time()
            t45=t5-t4
            if x[0]> -1000:
                m_list.append(m_index)
                bisection_list.append(x)
                flat_number+=1
            if flat_number ==action_number:
                break
            list_in.pop(node_to_trans)
        t6=time.time()
        t63=t6-t3
        return m_list,bisection_list
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his))*self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()


