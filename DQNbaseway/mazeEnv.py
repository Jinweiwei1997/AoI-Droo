# 构造gym Maze环境
import math
import random

import gym
from gym import spaces
import numpy as np
import time
import scipy.io as sio

from gym.envs.classic_control import rendering



class Maze(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    # 将会初始化动作空间与状态空间，便于强化学习算法在给定的状态空间中搜索合适的动作
    # 环境中会用的全局变量可以声明为类（self.）的变量
    def __init__(self):
        self.action_space = spaces.Discrete(4)  # 0全体吸收能量，1-5分别为几个节点吸收能量
        self.observation_space = spaces.Box(np.array([0,0,0,1,0,0,0,1,0,0,0,1]),\
                                            np.array(([4, 4, 3, 4,4,4,3, 4,4, 4,3,4])), dtype=np.int)
        self.n_actions = self.action_space.n
        self.n_states = self.observation_space.shape[0] # 转态向量维度
        self.state = None
        '''self.target = {(4,2): 50}   # 安全/目标状态
        self.danger = {(2,2): -20, (3,3): -20}  # 危险状态
        '''
        self.fixState=[1,2,1,1,1,2,0,1,2,2,2,1]
        data = sio.loadmat('./data/data_5')
        channel_h = data['input_h']
        channel_g = sio.loadmat('./data/data_5')['input_g']
    # 接收一个动作，执行这个动作
    # 用来处理状态的转换逻辑
    # 返回动作的回报、下一时刻的状态、以及是否结束当前episode及调试信息
    def step(self, action,number=0,Mode='train',):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        fi=[0,0.22314,0.510826,0.91629,1.609]
        B_max = 3 * 10 ** -4
        B_min = 0
        A_max: int = 4
        h = []  # 上行链路信道增益
        g = []  # 下行链路信道增益
        BEnergy = []  # 电池能量
        AoI = []
        sigma = 3.162277660168375 * 10 ** (-13)
        S = 15
        AverSumAoI = 0  # Sum of AoI at base station
        theta = []  # never used 权重
        eta = 0.5  # gain loss
        P = 5.012
        EnergyHarvest= [0,0,0,0,0]

        h_now = []
        g_now = []
        BEnergy_now = []
        AoI_now = []
        for i in range(3):  # 3个用户的数据要遍历
            if i == 0:
                d = 20
            if i == 1:
                d = 25
            if i == 2:
                d = 40
            # 四个变量要遍历
            h_k = 0.2*fi[self.state[4*i]]/d/d
            g_k = 0.2 * fi[self.state[4 * i+1]] / d / d
            BEnergy_a = B_min + (B_max - B_min) / 3 * self.state[4*i+2]
            h.append(h_k)
            g.append(g_k)
            BEnergy.append(BEnergy_a)
            AoI.append(self.state[4*i+3])
        AoI_k = [x for x in AoI]
        BEnergy_k = [x for x in BEnergy]
        if action == 0:  # 吸收能量
            for j in range(3):
                # calculate the Energy Harvested
                EnergyHarvest[j] = eta * P * g[j]
                B_next = BEnergy_k[j] + EnergyHarvest[j]
                if B_next >= B_max:
                    BEnergy_k[j] = B_max
                else:
                    BEnergy_k[j] += EnergyHarvest[j]
                # calculate the Sum of AoI
            for j in range(3):
                if (AoI[j] < A_max):
                    AoI_k[j] = AoI[j] + 1
                else:
                    AoI_k[j] = A_max
            for j in range(3):
                AverSumAoI += AoI_k[j]
            AverSumAoI /= (3)
        elif action == 1:  # 第一个节点发送数据包
            if h[0]==0:
                EnergyTrans = 100000
            else:
                EnergyTrans = sigma / h[0] * (2 ** S)

            for j in range(3):
                if (AoI[j] < A_max and j!=0):  #第一个节点发送
                    AoI_k[j] = AoI[j] + 1
                else:
                    AoI_k[j] = A_max
            AoI_k[0] = 1
            if BEnergy[0]<EnergyTrans:
                BEnergy_k[0] = -100 #如果是采取了动作变成负数了，那就变-1000
            else:
                BEnergy_k[0] -= EnergyTrans
        elif action == 2:  # 下
            if h[1]==0:
                EnergyTrans=100000
            else:
                EnergyTrans = sigma / h[1] * (2 ** S)

            for j in range(3):
                if (AoI[j] < A_max and  j!=1):
                    AoI_k[j] = AoI[j] + 1
                else:
                    AoI_k[j] = A_max
            AoI_k[1] = 1
            for j in range(3):
                AverSumAoI += AoI_k[j]
            AverSumAoI /= (3)
            if BEnergy[1] < EnergyTrans:
                BEnergy_k[1] = -100 # 如果是采取了动作变成负数了，那就变-1000
            else:
                BEnergy_k[1] -= EnergyTrans
        elif action == 3:  # 左
            if h[2]==0:
                EnergyTrans = 100000
            else:
                EnergyTrans = sigma / h[2] * (2 ** S)

            for j in range(3):
                if (AoI[j] < A_max and j != 2):
                    AoI_k[j] = AoI[j] + 1
                else:
                    AoI_k[j] = A_max
            AoI_k[2] = 1
            for j in range(3):
                AverSumAoI += AoI_k[j]
            AverSumAoI /= 3
            if BEnergy[2] < EnergyTrans:
                BEnergy_k[2] = -10000  # 如果是采取了动作变成负数了，那就变-1000
            else:
                BEnergy_k[2] -= EnergyTrans
        '''
        elif action == 4:  # 右

            EnergyTrans = sigma / h[3] * (2 ** S)
            for j in range(3):
                if (AoI[j] < A_max and j != 3):
                    AoI_k[j] = AoI[j] + 1
                else:
                    AoI_k[j] = A_max
            AoI_k[3] = 1
            for j in range(3):
                AverSumAoI += AoI_k[j]
            AverSumAoI /= (3)
            if BEnergy[3] < EnergyTrans:
                BEnergy_k[3] = BEnergy_k[3]  # 如果是采取了动作变成负数了，那就变-1000
            else:
                BEnergy_k[3] -= EnergyTrans
        elif action == 5:
            EnergyTrans = sigma / h[4] * (2 ** S)

            for j in range(5):
                if (AoI[j] < A_max and j != 4):
                    AoI_k[j] = AoI[j] + 1
                else:
                    AoI_k[j] = A_max
            AoI_k[4] = 1
            for j in range(5):
                AverSumAoI += AoI_k[j]
            AverSumAoI /= 5
            AssertionError(AverSumAoI<=6)
            if BEnergy[4] < EnergyTrans:
                BEnergy_k[4] = BEnergy_k[4]  # 如果是采取了动作变成负数了，那就变-1000
            else:
                BEnergy_k[4] -= EnergyTrans
'''
        for i in range(3):
            #h_now.append(int((h[i] - h_min)/((h_max-h_min)/5)))
            #g_now.append(int((g[i] - h_min)/((h_max-h_min)/5)))
            if i==0:
                d=20
            if i==1:
                d=25
            if i==2:
                d=40
            if Mode=='train':  #训练的时候存数据
                H=random.expovariate(1)
                G=random.expovariate(1)
                flat1=0
                flat2=0
                for j in range(len(fi)-1):
                    if fi[j]<=H<fi[j+1]:
                        flat1=1
                        h_index =j
                        break
                if flat1==0:
                    h_index=4
                h_now.append(h_index)
                for j in range(len(fi) - 1):
                    if fi[j] <= G < fi[j + 1]:
                        flat2 = 1
                        g_index = j
                        break
                if flat2 == 0:
                    g_index = 4
                g_now.append(g_index)
            else:
                H = random.expovariate(1)
                G = random.expovariate(1)
                flat1 = 0
                flat2 = 0
                for j in range(len(fi) - 1):
                    if fi[j] <= H < fi[j + 1]:
                        flat1 = 1
                        h_index = j
                        break
                if flat1 == 0:
                    h_index = 4
                h_now.append(h_index)
                for j in range(len(fi) - 1):
                    if fi[j] <= G < fi[j + 1]:
                        flat2 = 1
                        g_index = j
                        break
                if flat2 == 0:
                    g_index = 4
                g_now.append(g_index)
            B_int = int((BEnergy_k[i]-B_min)/((B_max-B_min)/3))
            if B_int >3:
                BEnergy_now.append(3)
            else:
                BEnergy_now.append(B_int)
        next_index=[]
        for i in range(3):
            next_index.append(h_now[i])
            next_index.append(g_now[i])
            next_index.append(BEnergy_now[i])
            next_index.append(AoI_k[i])
        next_state = np.array(next_index)
        flat=0    #标记是否电池过量
        done = False   #默认状态是否不对
        reward = 0     #默认reward=0
        for i in range(3):
            if next_state[4*i+2] <0:
                done = False
                reward = 100
                AoI_k=[x for x in AoI]
                flat = 1
                break

        if flat == 0:
            reward =0
            for i in range(3):
                reward+=AoI_k[i]
            reward /= 3
            done = True
            self.state = next_state
            AssertionError(reward<=4)

        self.counts += 1
        return self.state, -reward, done,AoI_k

    # 用于在每轮开始之前重置智能体的状态，把环境恢复到最开始
    def reset(self, startstate=None):
        '''
        :param startstate: (1,1)
        :return:
        '''
        #if startstate==None:
        self.state = self.observation_space.sample()
        #else:
         #   self.state = startstate
        self.counts = 0
        return self.state

    # metadata、render()、close()是与图像显示有关的，我们不涉及这一部分，感兴趣的同学可以自行编写相关内容。
    # render()绘制可视化环境的部分都写在这里
    '''def render(self, mode='human'):
        # 绘制网格
        for i in range(5):
            # 竖线
            self.viewer.draw_line(
                (50, 50),
                (50, 450),
                color=(0, 0, 0)
            ).add_attr(rendering.Transform((100 * i, 0)))
            # 横线
            self.viewer.draw_line(
                (50, 50), (450, 50)).add_attr(rendering.Transform((0, 100 * i)))

        # 绘制出口（安全状态）
        for state in self.target:
            self.drawrectangle2(state, color=(0, 1, 0))

        # 绘制危险区域
        for state in self.danger:
            self.drawrectangle2(state, color=(1, 0, 0))

        # 绘制当前state的位置(圆)
        size = 100
        center = (
            50 + self.state[0] * size - 0.5 * size,
            50 + self.state[1] * size - 0.5 * size)
        self.viewer.draw_circle(
            48, 30, filled=True, color=(1, 1, 0)).add_attr(rendering.Transform(center))

        return self.viewer.render(return_rgb_array=mode == 'human')
'''





if __name__ == '__main__':
    env = Maze()
    for epoch in range(100):
        env.reset()
        while True:
            print(env.state)
            #env.render()
            action = env.action_space.sample()
            print(action)
            x=env.step(action)[0]
            print(x)

            time.sleep(0.5)
            break
    #env.close()
